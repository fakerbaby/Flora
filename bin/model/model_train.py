# -*- coding: utf-8 -*-
# @File : TrainModel.py
# @Author : Zhengming Li
# @Date : Nov 2021
# @Description ：训练 model.py 中的 GAT_FeatureToDistance 和 GAT_DistanceToPosition
import os
import sys
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))  # the absolute path to base directory
sys.path.append(BASE_DIR)

from bin.model.load_data import read_data
from bin.model.model import GAT_DistanceToPosition, model1, ResGAT_Distance, ResGAT_Position, ResGAT_Distance_noMLP, \
    ResGAT_Position_noMLP
from torch import nn, optim
import torch
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

BENCHMARK = 'ispd2005'
# DATASET_NAME = 'first'
# DATASET_NAME = 'adaptec1'
# DATASET_NAME = 'adaptec1-v1'
DATASET_NAME = 'adaptec1'
# DATASET_NAME = 'bigblue1'
# DATASET_NAME = 'industry_625@zhouhai'

# MODEL_NAME = 'twolayerGAT'
MODEL_NAME = 'ResGAT'
# MODEL_NAME = 'ResGAT_noMLP'

LODE_NAME = 'adaptec1_576'
# LODE_NAME = 'feature650'
# LODE_NAME = 'feature625'

DATA_FILE_NAME = 'data/' + BENCHMARK + DATASET_NAME
MODEL_FILE_PATH = 'model/' + MODEL_NAME + '/' + DATASET_NAME
RESULT_FILE_PATH = 'result/' + MODEL_NAME + '/' + DATASET_NAME

# NODE_NUMBER = 650
GROUP_NUM = 625
# NODE_NUMBER = 625
ITERATION = 20

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

LR1 = 1e-3
LR2 = 1e-3


def distance_calculation(pre, adj, num):
    s = torch.ones(num, num)
    s = s.to(DEVICE)
    x0 = s * pre[:, 0]
    x1 = x0.T
    X = (x0 - x1) ** 2
    y0 = s * pre[:, 1]
    y1 = y0.T
    Y = (y0 - y1) ** 2

    pre_dis_array = X + Y

    mask_dis = torch.mul(pre_dis_array, adj)
    # mask_dis = mask_dis.to(device)
    return mask_dis


def shuffle_the_data(N, i, feature, adj, label, distance, device):
    """
    shuffle the data and then return
    Args:
        N [int]: batch size
        i [int]: number of batch
        feature [narray]: vector of feature
        adj [narray]: adjacency matrix
        label [narray]: vector of label
        distance [narray]: vector of distance
        device [String]: device's name (cuda:3 or cpu)
    Return:
        Feature, Distance, Adj, Label
    """

    # 取出一个 batch
    Label = label[N * i: N * (i + 1)][:]
    Adj = adj[N * i: N * (i + 1)][:]
    Feature = feature[N * i: N * (i + 1)][:]
    Distance = distance[N * i: N * (i + 1)][:]

    # 打乱顺序
    init_order = list(range(0, len(Feature)))
    random_order = list(range(0, len(Feature)))
    random.shuffle(random_order)
    # change the row order
    Label[init_order] = Label[random_order]
    Feature[init_order] = Feature[random_order]
    Adj[init_order] = Adj[random_order]
    Distance[init_order] = Distance[random_order]
    # change the column order numpy切片参考：https://blog.csdn.net/qq_30835655/article/details/71055198
    Feature[:, [init_order]] = Feature[:, [random_order]]
    Adj[:, [init_order]] = Adj[:, [random_order]]
    Distance[:, [init_order]] = Distance[:, [random_order]]

    # covert narray to tensor
    Label = torch.from_numpy(Label).float().to(device)
    Feature = torch.from_numpy(Feature).float().to(device)
    Distance = torch.from_numpy(Distance).float().to(device)
    Adj = torch.from_numpy(Adj).float().to(device)

    return Feature, Adj, Label, Distance


class ModelTrain:
    def __init__(self):
        # test dataset
        self.test_adj = None
        self.test_distance = None
        self.test_feature = None
        self.test_label = None
        # train dataset
        self.train_adj = None
        self.train_distance = None
        self.train_feature = None
        self.train_label = None
        # final dataset
        self.final_adj = None
        self.final_distance = None
        self.final_feature = None

        self.my_order = None
        self.random_order = None

    def load_data(self, data_file_path):
        self.test_feature, self.test_adj, self.test_label = read_data(
            '%s/test_label.csv' % data_file_path)
        self.test_distance = np.loadtxt(
            '%s/test_distance.csv' % data_file_path, delimiter=',')
        print("test data ready")
        self.train_feature, self.train_adj, self.train_label = read_data(
            '%s/train_label.csv' % data_file_path)
        self.train_distance = np.loadtxt(
            '%s/train_distance.csv' % data_file_path, delimiter=',')
        print("train data ready")
        # return test_adj, test_distance, test_feature, test_label, train_adj, train_distance, train_feature, train_label

    def load_final_data(self, data_file_path):
        self.final_feature = np.loadtxt('%s/final_feature.csv' %
                                        data_file_path, delimiter=",")
        self.final_distance = np.loadtxt('%s/final_distance.csv' %
                                         data_file_path, delimiter=",")
        self.final_adj = np.zeros(
            shape=(len(self.final_feature), len(self.final_feature[0])), dtype="double")
        # connectivity_index is a tuple data, each element is an array
        connectivity_index = np.where(self.final_feature > 0)
        for i, j in zip(connectivity_index[0], connectivity_index[1]):
            self.final_adj[i][j] = 1
        for k in range(len(self.final_feature[0])):
            self.final_adj[k][k] = 1
        self.final_feature = torch.from_numpy(self.final_feature).float().to(DEVICE)
        self.final_distance = torch.from_numpy(self.final_distance).float().to(DEVICE)
        self.final_adj = torch.from_numpy(self.final_adj).float().to(DEVICE)

        self.my_order = list(range(0, len(self.final_feature)))
        self.random_order = list(range(0, len(self.final_feature)))
        random.shuffle(self.random_order)

        # change the row order
        self.final_feature[self.my_order] = self.final_feature[self.random_order]
        self.final_distance[self.my_order] = self.final_distance[self.random_order]
        self.final_adj[self.my_order] = self.final_adj[self.random_order]
        # label[my_order] = label[random_order]

    def load_final_data_without_distance(self, data_file_path):
        self.final_feature = np.loadtxt('%s/%s.csv' %
                                        (data_file_path, LODE_NAME), delimiter=",")
        self.final_adj = np.zeros(
            shape=(len(self.final_feature), len(self.final_feature[0])), dtype="double")
        # connectivity_index is a tuple data, each element is an array
        connectivity_index = np.where(self.final_feature > 0)
        for i, j in zip(connectivity_index[0], connectivity_index[1]):
            self.final_adj[i][j] = 1
        for k in range(len(self.final_feature[0])):
            self.final_adj[k][k] = 1
        self.final_feature = torch.from_numpy(self.final_feature).float().to(DEVICE)
        self.final_adj = torch.from_numpy(self.final_adj).float().to(DEVICE)
        self.my_order = list(range(0, len(self.final_feature)))
        self.random_order = list(range(0, len(self.final_feature)))
        random.shuffle(self.random_order)
        # change the row order
        self.final_feature[self.my_order] = self.final_feature[self.random_order]
        self.final_adj[self.my_order] = self.final_adj[self.random_order]
        # label[my_order] = label[random_order]
        # return adj_matrix, feature, my_order, random_order

    # def load_final_data_without_distance_and_shuffle(self, data_file_path):
    #     feature = np.loadtxt('%s/%s.csv' %
    #                          (data_file_path, LODE_NAME), delimiter=",")
    #     adj_matrix = np.zeros(
    #         shape=(len(feature), len(feature[0])), dtype="double")
    #     # connectivity_index is a tuple data, each element is an array
    #     connectivity_index = np.where(feature > 0)
    #     for i, j in zip(connectivity_index[0], connectivity_index[1]):
    #         adj_matrix[i][j] = 1
    #     for k in range(len(feature[0])):
    #         adj_matrix[k][k] = 1
    #     feature = torch.from_numpy(feature).float().to(DEVICE)
    #     adj_matrix = torch.from_numpy(adj_matrix).float().to(DEVICE)

    #     return adj_matrix, feature

    def loss_func(self, prediction_position, label_position, adj_matrix, node_num, train_distance):
        criterion = nn.MSELoss()
        loss1 = criterion(prediction_position, label_position)
        # print("test distance loss1 :", loss1)  # distance between pre_pos and label_pos

        pre_dis = distance_calculation(prediction_position, adj_matrix, node_num)
        mask_label = torch.mul(train_distance, adj_matrix)

        print(torch.sum(pre_dis))
        print(torch.sum(mask_label))
        loss2 = criterion(pre_dis, mask_label)
        loss2 = loss2.cpu()

        # print("loss2: ", loss2)  # total wirelength
        # loss_total = loss1 + loss2
        # loss_total = loss2
        # print("total_loss: ", loss_total)

        # return loss_total
        return loss1 + loss2

    def restore_order_and_save_result(self, prediction_distance, prediction_position):
        distance_mask_sum = torch.mul(self.final_distance, self.final_adj).sum()
        prediction_distance_mask_sum = torch.mul(
            prediction_distance, self.final_adj).sum()
        ods = (prediction_distance_mask_sum - distance_mask_sum) / \
              distance_mask_sum * 100
        print("***************************************************************")
        print("***************************************************************")
        print("***************************************************************")
        print("***                                                         ***")
        print("***                                                         ***")
        print("***                                                         ***")
        print("***                    ods = %.2f %%                        ***" %
              ods.item())  # 输出distance之间的差值
        print("***                                                         ***")
        print("***                                                         ***")
        print("***                                                         ***")
        print("***************************************************************")
        print("***************************************************************")
        print("***************************************************************")
        final_order = []
        for i in range(len(self.random_order)):
            final_order.append(self.random_order.index(i))
        prediction_position[self.my_order] = prediction_position[final_order]
        result_file_name = '%s/result.csv' % RESULT_FILE_PATH
        if os.path.exists(result_file_name):
            os.remove(result_file_name)
        os.system("mkdir -p %s" % (os.path.dirname(result_file_name)))
        np.savetxt(result_file_name, prediction_position, delimiter=',')
        x = prediction_position[:, 0]
        y = prediction_position[:, 1]
        plt.scatter(x, y)
        pic_file_name = '%s/out.png' % RESULT_FILE_PATH
        print(pic_file_name)
        if os.path.exists(pic_file_name):
            os.remove(pic_file_name)
        os.system("mkdir -p %s" % (os.path.dirname(pic_file_name)))
        plt.savefig(pic_file_name)

    def restore_order_and_save_result_without_distance(self, prediction_position):
        final_order = []
        for i in range(len(self.random_order)):
            final_order.append(self.random_order.index(i))
        prediction_position[self.my_order] = prediction_position[final_order]
        result_file_name = '%s/result_%s.csv' % (RESULT_FILE_PATH, LODE_NAME)
        if os.path.exists(result_file_name):
            os.remove(result_file_name)
        os.system("mkdir -p %s" % (os.path.dirname(result_file_name)))
        np.savetxt(result_file_name, prediction_position, delimiter=',')
        x = prediction_position[:, 0]
        y = prediction_position[:, 1]
        plt.scatter(x, y)
        # plt.xlim((5, 15))
        pic_file_name = '%s/out_%s.png' % (RESULT_FILE_PATH, LODE_NAME)
        if os.path.exists(pic_file_name):
            os.remove(pic_file_name)
        os.system("mkdir -p %s" % (os.path.dirname(pic_file_name)))
        plt.savefig(pic_file_name)

    # def restore_order_and_save_result_without_distance_and_shuffle(prediction_position):
    #     result_file_name = '%s/result_%s_NoShuffle.csv' % (
    #         RESULT_FILE_PATH, LODE_NAME)
    #     if os.path.exists(result_file_name):
    #         os.remove(result_file_name)
    #     os.system("mkdir -p %s" % (os.path.dirname(result_file_name)))
    #     np.savetxt(result_file_name, prediction_position, delimiter=',')
    #     x = prediction_position[:, 0]
    #     y = prediction_position[:, 1]
    #     plt.scatter(x, y)
    #     pic_file_name = '%s/out_%s_NoShuffle.png' % (
    #         RESULT_FILE_PATH, LODE_NAME)
    #     if os.path.exists(pic_file_name):
    #         os.remove(pic_file_name)
    #     os.system("mkdir -p %s" % (os.path.dirname(pic_file_name)))
    #     plt.savefig(pic_file_name)

    def train_and_test_distance_model(self):
        print("******************** Preparing Data ********************")
        # distanceModel = GAT_FeatureToDistance(len(train_feature[0]))
        distanceModel = model1(len(self.train_feature[0]), int(2), int(2))

        # if torch.cuda.device_count() > 1:
        #     distanceModel = nn.DataParallel(distanceModel)
        distanceModel.to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(distanceModel.parameters(), lr=1e-3)
        max_iterations = 500

        flag = 0  # 用于test，得到最好模型
        best_loss = float("inf")  # 正无穷
        print("****************************** Training Distance Model ******************************")
        for iteraiton in range(max_iterations):
            running_loss = 0
            # batch size = len(train_feature) // NODE_NUMBER
            for i in range(len(self.train_feature) // GROUP_NUM):
                train_Feature, train_Adj, train_Label, train_Distance = shuffle_the_data(GROUP_NUM, i,
                                                                                         self.train_feature,
                                                                                         self.train_adj,
                                                                                         self.train_label,
                                                                                         self.train_distance,
                                                                                         DEVICE)
                # training the GAT_FeauterToDistance model
                first_GAT_output = distanceModel(
                    train_Feature, train_Adj)

                try:
                    loss = criterion(torch.mul(first_GAT_output, train_Adj), torch.mul(
                        train_Distance, train_Adj))
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('| WARNING2: ran out of memory')
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise e
                if (i + 1) % 20 == 0:
                    print("[Train Distance Model epoch %03d, batch %02d] loss: %.3f" % (
                        iteraiton + 1, i + 1, loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            # print("[epoch %03d] loss: %.3f" % (iteraiton + 1, loss))
            # testing the model

            total_loss_test, final_loss = 0, 0
            with torch.no_grad():
                # determine how many graphs in test data set
                for i in range(len(self.test_feature) // GROUP_NUM):
                    F, Adj, L, D = shuffle_the_data(
                        GROUP_NUM, i, self.test_feature, self.test_adj, self.test_label, self.test_distance,
                        DEVICE)  # shuffle the data
                    embedding = distanceModel(F, Adj)
                    criterion = nn.MSELoss()
                    try:
                        ave_loss = criterion(torch.mul(embedding, Adj),
                                             torch.mul(D, Adj))  # lost on per-chip22   q
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print('| WARNING4: ran out of memory')
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                        else:
                            raise e
                total_loss_test += ave_loss.item()  # accumulate the loss on all chips all nodes

            print("[Testing epoch %03d] loss: %.3f" % (
            iteraiton + 1, total_loss_test / (len(self.test_feature) // GROUP_NUM)))
            if total_loss_test < best_loss:
                best_loss = total_loss_test
                print("Saving the model and zero the flag ")

                model_file_name = '%s/distance_model.pt' % MODEL_FILE_PATH
                if os.path.exists(model_file_name):
                    os.remove(model_file_name)
                os.system("mkdir -p %s" % (os.path.dirname(model_file_name)))
                torch.save(distanceModel.state_dict(), model_file_name)
                print("[best loss] =", best_loss / (len(self.test_feature) // GROUP_NUM))  # 平均一张图的所有macro损失
                flag = 0
            else:
                flag += 1
                print("flag = %d" % flag)
                if flag == ITERATION:
                    print("[best loss] =", best_loss / (len(self.test_feature) // GROUP_NUM))  # 平均一张图的所有macro损失
                    print("******************** End of Training ********************")
                    break

            print("************************************************************")

    def train_and_test_position_model(self):
        compact = 2  # the dimension of GAT compacted feature
        # distance_model = GAT_FeatureToDistance(len(train_feature[0]))
        distance_model = model1(len(self.train_feature[0]), int(2), int(2))

        position_model = GAT_DistanceToPosition(len(self.train_distance[0]), int(compact))
        # if torch.cuda.device_count() > 1:
        #     position_model = nn.DataParallel(position_model)
        position_model.to(DEVICE)
        print('%s/distance_model.pt' % MODEL_FILE_PATH)
        distance_model.load_state_dict(torch.load('%s/distance_model.pt' % MODEL_FILE_PATH))
        distance_model.to(DEVICE)
        distance_model.eval()

        optimizer = optim.Adam(position_model.parameters(), lr=1e-4)
        max_iterations = 500

        flag = 0  # 用于test，得到最好模型
        best_loss = float("inf")  # 正无穷
        print("******************** Training Position Model ********************")
        for iteraiton in range(max_iterations):
            running_loss = 0
            # batch size = len(train_feature) // NODE_NUMBER
            for i in range(len(self.train_feature) // GROUP_NUM):
                train_Feature, train_Adj, train_Label, train_Distance = shuffle_the_data(GROUP_NUM, i,
                                                                                         self.train_feature,
                                                                                         self.train_adj,
                                                                                         self.train_label,
                                                                                         self.train_distance,
                                                                                         DEVICE)

                prediction_distance = distance_model(train_Feature, train_Adj)
                second_model_output = position_model(prediction_distance, train_Adj)
                try:
                    loss = self.loss_func(second_model_output, train_Label,
                                          train_Adj, GROUP_NUM, train_Distance)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('| WARNING2: ran out of memory')
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise e

                if (i + 1) % 20 == 0:
                    print("[Train Position Model epoch %03d, batch %02d] loss: %.3f" % (iteraiton + 1, i + 1, loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            total_loss_test, final_loss = 0, 0
            with torch.no_grad():
                # determine how many graphs in test data set
                for i in range(len(self.test_feature) // GROUP_NUM):
                    F, Adj, L, D = shuffle_the_data(
                        GROUP_NUM, i, self.test_feature, self.test_adj, self.test_label, self.test_distance,
                        DEVICE)  # shuffle the data
                    prediction_distance = distance_model(F, Adj)
                    embedding = position_model(prediction_distance, Adj)

                    try:
                        ave_loss = self.loss_func(embedding, L, Adj, GROUP_NUM, D)
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print('| WARNING4: ran out of memory')
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                        else:
                            raise e

                    total_loss_test += ave_loss.item()  # accumulate the loss on all chips all nodes

            print("[Testing epoch %03d] loss: %.3f" % (iteraiton + 1,
                                                       total_loss_test / (len(self.test_feature) // GROUP_NUM)))
            if total_loss_test < best_loss:
                best_loss = total_loss_test

                print("Saving the model and zero the flag ")
                model_file_name = '%s/position_model.pt' % MODEL_FILE_PATH
                if os.path.exists(model_file_name):
                    os.remove(model_file_name)
                os.system("mkdir -p %s" % (os.path.dirname(model_file_name)))
                torch.save(position_model.state_dict(), model_file_name)
                print("best loss=", best_loss /
                      (len(self.test_feature) // GROUP_NUM))  # 平均一张图的所有macro损失
                flag = 0
            else:
                flag += 1
                print("flag = %d" % flag)
                if flag == ITERATION:
                    print("best loss=", best_loss /
                          (len(self.test_feature) // GROUP_NUM))  # 平均一张图的所有macro损失
                    print("******************** End of Training ********************")
                    break

            print("************************************************************")

    def load_two_layer_model(self):
        # distance_model
        distance_model = model1(len(self.final_feature[0]), int(2), int(2)).to(DEVICE)
        distance_model.load_state_dict(torch.load('%s/distance_model.pt' % MODEL_FILE_PATH))
        distance_model.eval()
        # position_model
        position_model = GAT_DistanceToPosition(
            len(self.final_feature[0]), int(2)).to(DEVICE)
        position_model.load_state_dict(torch.load('%s/position_model.pt' % MODEL_FILE_PATH))
        position_model.eval()

        prediction_distance = distance_model(self.final_feature, self.final_adj)
        prediction_position = position_model(prediction_distance, self.final_adj)
        prediction_position = prediction_position.cpu()
        prediction_position = prediction_position.detach().numpy()

        self.restore_order_and_save_result(prediction_distance, prediction_position)

    # ResGAT
    def train_res_gat_model(self):
        model_res_distance = ResGAT_Distance(
            len(self.train_feature[0]), int(2), int(len(self.train_feature[0]) / 2))
        model_res_distance.to(DEVICE)
        model_res_position = ResGAT_Position(
            len(self.train_feature[0]), int(2), int(len(self.train_feature[0]) / 2), int(2))
        model_res_position.to(DEVICE)

        criterion_distance = nn.MSELoss()
        optimizer_distance = optim.Adam(
            model_res_distance.parameters(), lr=LR1)
        optimizer_position = optim.Adam(
            model_res_position.parameters(), lr=LR2)

        max_iterations = 500
        flag = 0  # 用于test，得到最好模型
        best_loss = float("inf")  # 正无穷

        for iteraiton in range(max_iterations):
            running_loss_distance = 0
            running_loss_position = 0

            # batch size = len(train_feature) // GROUP_NUM
            for i in range(len(self.train_feature) // GROUP_NUM):
                train_Feature, train_Adj, train_Label, train_Distance = shuffle_the_data(GROUP_NUM, i,
                                                                                         self.train_feature,
                                                                                         self.train_adj,
                                                                                         self.train_label,
                                                                                         self.train_distance,
                                                                                         DEVICE)  # load data

                train_Feature_pos = train_Feature
                train_Adj_pos = train_Adj
                train_Label_pos = train_Label
                train_Distance_pos = train_Distance

                # detect the inplace operator
                # torch.autograd.set_detect_anomaly(True)

                # 训练 ResGAT_Distance
                _, prediction_distance = model_res_distance(
                    train_Feature, train_Adj)  # forward
                loss_distance = criterion_distance(torch.mul(prediction_distance, train_Adj),
                                                   torch.mul(train_Distance, train_Adj))
                optimizer_distance.zero_grad()
                # 一共两个loss，不加括号内的会报错，下一个 backward 会覆盖掉这的值，从而下次训练读取不到正确的数据
                loss_distance.backward(retain_graph=True)
                optimizer_distance.step()
                # running_loss_distance += loss_distance.item()

                # 训练 ResGAT_Position
                distance_gat_output, _ = model_res_distance(
                    train_Feature_pos, train_Adj_pos)
                prediction_position = model_res_position(
                    train_Feature_pos, distance_gat_output, train_Adj_pos)
                loss_position = self.loss_func(prediction_position, train_Label_pos, train_Adj_pos, GROUP_NUM,
                                               train_Distance_pos)
                optimizer_position.zero_grad()
                # with torch.autograd.detect_anomaly():
                loss_position.backward()
                optimizer_position.step()
                # running_loss_position += loss_position.item()

                if (i + 1) % 20 == 0:
                    print("[Epoch %03d, batch %02d] loss_distance: %.3f, loss_position: %.3f" % (
                        iteraiton + 1, i + 1, loss_distance, loss_position))
            # testing the model

            total_loss_test, final_loss = 0, 0
            with torch.no_grad():
                # determine how many graphs in test data set
                for i in range(len(self.test_feature) // GROUP_NUM):
                    F, Adj, L, D = shuffle_the_data(
                        GROUP_NUM, i, self.test_feature, self.test_adj,
                        self.test_label, self.test_distance, DEVICE)  # shuffle the data
                    F_pos = F
                    Adj_pos = Adj
                    L_pos = L
                    D_pos = D
                    distance_gat_mid_output, prediction_distance = model_res_distance(
                        F, Adj)
                    pre_test_position = model_res_position(F_pos, distance_gat_mid_output, Adj_pos)
                    ave_loss = self.loss_func(pre_test_position, L_pos, Adj_pos, GROUP_NUM, D_pos)
                    total_loss_test += ave_loss.item()  # accumulate the loss on all chips all nodes

            print("[Testing epoch %03d] loss: %.3f" % (iteraiton + 1,
                                                       total_loss_test / (len(self.test_feature) // GROUP_NUM)))

            if total_loss_test < best_loss:
                best_loss = total_loss_test
                print("Saving the model and zero the flag ")

                model_file_name = '%s/ResGAT_model_distance.pt' % MODEL_FILE_PATH
                if os.path.exists(model_file_name):
                    os.remove(model_file_name)
                os.system("mkdir -p %s" % (os.path.dirname(model_file_name)))
                torch.save(model_res_distance.state_dict(), model_file_name)
                model_file_name = '%s/ResGAT_model_position.pt' % MODEL_FILE_PATH

                if os.path.exists(model_file_name):
                    os.remove(model_file_name)
                os.system("mkdir -p %s" % (os.path.dirname(model_file_name)))
                torch.save(model_res_position.state_dict(), model_file_name)

                print("best loss=", best_loss /
                      (len(self.test_feature) // GROUP_NUM))  # 平均一张图的所有macro损失
                flag = 0
            else:
                flag += 1
                print("flag = %d" % flag)
                if flag == ITERATION:
                    print("best loss=", best_loss /
                          (len(self.test_feature) // GROUP_NUM))  # 平均一张图的所有macro损失
                    print("******************** End of Training ********************")
                    break

            print("************************************************************")

    def test_res_gat_model(self, model_res_distance, model_res_position, feature, adj, label, N, dis):
        s, final_loss = 0, 0
        with torch.no_grad():
            # determine how many graphs in test data set
            for i in range(len(feature) // GROUP_NUM):
                F, Adj, L, D = shuffle_the_data(
                    GROUP_NUM, i, feature, adj, label, dis, DEVICE)  # shuffle the data
                F_pos = F
                Adj_pos = Adj
                L_pos = L
                D_pos = D
                distance_gat_mid_output, prediction_distance = model_res_distance(
                    F, Adj)
                pre = model_res_position(
                    F_pos, distance_gat_mid_output, Adj_pos)
                ave_loss = loss_func(pre, L_pos, Adj_pos, GROUP_NUM, D_pos)
                s += ave_loss.item()  # accumulate the loss on all chips all nodes

        return s, pre

    def load_res_model(self):
        self.final_feature[:, [self.my_order]] = self.final_feature[:, [self.random_order]]
        self.final_distance[:, [self.my_order]] = self.final_distance[:, [self.random_order]]
        self.final_adj[:, [self.my_order]] = self.final_adj[:, [self.random_order]]

        model_res_distance = ResGAT_Distance(
            len(self.final_feature[0]), int(2), int(len(self.final_feature[0]) / 2))
        model_res_distance.to(DEVICE)
        model_res_distance.load_state_dict(torch.load(
            '%s/ResGAT_model_distance.pt' % MODEL_FILE_PATH))
        model_res_distance.eval()

        model_res_position = ResGAT_Position(
            len(self.final_feature[0]), int(2), int(len(self.final_feature[0]) / 2), int(2))
        model_res_position.to(DEVICE)
        model_res_position.load_state_dict(torch.load(
            '%s/ResGAT_model_position.pt' % MODEL_FILE_PATH))
        model_res_position.eval()

        feature_pos = self.final_feature
        adj_matrix_pos = self.final_adj

        distance_gat_output, prediction_distance = model_res_distance(
            self.final_feature, self.final_adj)
        prediction_position = model_res_position(
            feature_pos, distance_gat_output, adj_matrix_pos)
        prediction_position = prediction_position.cpu()
        prediction_position = prediction_position.detach().numpy()

        self.restore_order_and_save_result(prediction_distance, prediction_position)

    def load_res_model_without_distance(self):
        self.final_feature[:, [self.my_order]] = self.final_feature[:, [self.random_order]]
        self.final_adj[:, [self.my_order]] = self.final_adj[:, [self.random_order]]
        model_res_distance = ResGAT_Distance(
            len(self.final_feature[0]), int(2), int(len(self.final_feature[0]) / 2))
        model_res_distance.to(DEVICE)

        model_res_distance.load_state_dict(torch.load(
            '%s/ResGAT_model_distance.pt' % MODEL_FILE_PATH))
        model_res_distance.eval()

        model_res_position = ResGAT_Position(
            len(self.final_feature[0]), int(2), int(len(self.final_feature[0]) / 2), int(2))
        model_res_position.to(DEVICE)

        model_res_position.load_state_dict(torch.load(
            '%s/ResGAT_model_position.pt' % MODEL_FILE_PATH))
        model_res_position.eval()

        feature_pos = self.final_feature
        adj_matrix_pos = self.final_adj
        distance_gat_output, prediction_distance = model_res_distance(
            self.final_feature, self.final_adj)
        prediction_position = model_res_position(
            feature_pos, distance_gat_output, adj_matrix_pos)
        prediction_position = prediction_position.cpu()
        prediction_position = prediction_position.detach().numpy()

        self.restore_order_and_save_result_without_distance(prediction_position)
        print("Ending")

    # def load_res_model_without_distance_and_shuffle():
    #     adj_matrix, feature = load_final_data_without_distance_and_shuffle()

    #     model_res_distance = ResGAT_Distance(
    #         len(feature[0]), int(2), int(len(feature[0]) / 2))
    #     model_res_distance.to(DEVICE)

    #     model_res_distance.load_state_dict(torch.load(
    #         '%s/ResGAT_model_distance.pt' % MODEL_FILE_PATH))
    #     model_res_distance.eval()

    #     model_res_position = ResGAT_Position(
    #         len(feature[0]), int(2), int(len(feature[0]) / 2), int(2))
    #     model_res_position.to(DEVICE)

    #     model_res_position.load_state_dict(torch.load(
    #         '%s/ResGAT_model_position.pt' % MODEL_FILE_PATH))
    #     model_res_position.eval()

    #     feature_pos = feature
    #     adj_matrix_pos = adj_matrix
    #     distance_gat_output, prediction_distance = model_res_distance(
    #         feature, adj_matrix)
    #     prediction_position = model_res_position(
    #         feature_pos, distance_gat_output, adj_matrix_pos)
    #     prediction_position = prediction_position.cpu()
    #     prediction_position = prediction_position.detach().numpy()

    #     restore_order_and_save_result_without_distance_and_shuffle(
    #         prediction_position)
    #     print("Ending")

    def train_res_gat_model_noMLP(self):
        model_res_distance = ResGAT_Distance_noMLP(
            len(self.train_feature[0]), int(2), int(len(self.train_feature[0]) / 2))
        model_res_distance.to(DEVICE)
        model_res_position = ResGAT_Position_noMLP(
            len(self.train_feature[0]), int(2), int(len(self.train_feature[0]) / 2), int(2))
        model_res_position.to(DEVICE)

        criterion_distance = nn.MSELoss()
        optimizer_distance = optim.Adam(
            model_res_distance.parameters(), lr=LR1)
        optimizer_position = optim.Adam(
            model_res_position.parameters(), lr=LR2)

        max_iterations = 500
        flag = 0  # 用于test，得到最好模型
        best_loss = float("inf")  # 正无穷

        for iteraiton in range(max_iterations):
            # batch size = len(train_feature) // N
            for i in range(len(self.train_feature) // GROUP_NUM):

                train_Feature, train_Adj, train_Label, train_Distance = shuffle_the_data(GROUP_NUM, i,
                                                                                         self.train_feature,
                                                                                         self.train_adj,
                                                                                         self.train_label,
                                                                                         self.train_distance,
                                                                                         DEVICE)  # load data

                train_Feature_pos = train_Feature
                train_Adj_pos = train_Adj
                train_Label_pos = train_Label
                train_Distance_pos = train_Distance

                # detect the inplace operator
                # torch.autograd.set_detect_anomaly(True)

                # 训练 ResGAT_Distance
                _, prediction_distance = model_res_distance(
                    train_Feature, train_Adj)  # forward
                loss_distance = criterion_distance(torch.mul(prediction_distance, train_Adj),
                                                   torch.mul(train_Distance, train_Adj))
                optimizer_distance.zero_grad()
                # 一共两个loss，不加括号内的会报错，下一个 backward 会覆盖掉这的值，从而下次训练读取不到正确的数据
                loss_distance.backward(retain_graph=True)
                optimizer_distance.step()

                # 训练 ResGAT_Position
                distance_gat_output, _ = model_res_distance(
                    train_Feature_pos, train_Adj_pos)
                prediction_position = model_res_position(
                    train_Feature_pos, distance_gat_output, train_Adj_pos)
                loss_position = self.loss_func(prediction_position, train_Label_pos, train_Adj_pos, GROUP_NUM,
                                               train_Distance_pos)
                optimizer_position.zero_grad()
                # with torch.autograd.detect_anomaly():
                loss_position.backward()
                optimizer_position.step()

                if (i + 1) % 20 == 0:
                    print("[Epoch %03d, batch %02d] loss_distance: %.3f, loss_position: %.3f" % (
                        iteraiton + 1, i + 1, loss_distance, loss_position))
                    running_loss_distance = 0
                    running_loss_position = 0

            # testing the model
            total_loss_test, final_loss = 0, 0
            with torch.no_grad():
                # determine how many graphs in test data set
                for i in range(len(self.test_feature) // GROUP_NUM):
                    F, Adj, L, D = shuffle_the_data(
                        GROUP_NUM, i, self.test_feature, self.test_adj,
                        self.test_label, self.test_distance, DEVICE)  # shuffle the data
                    F_pos = F
                    Adj_pos = Adj
                    L_pos = L
                    D_pos = D
                    distance_gat_mid_output, prediction_distance = model_res_distance(
                        F, Adj)
                    pre_test_position = model_res_position(F_pos, distance_gat_mid_output, Adj_pos)
                    ave_loss = self.loss_func(pre_test_position, L_pos, Adj_pos, GROUP_NUM, D_pos)
                    total_loss_test += ave_loss.item()  # accumulate the loss on all chips all nodes

            print("[Testing epoch %03d] loss: %.3f" % (iteraiton + 1,
                                                       total_loss_test / (len(self.test_feature) // GROUP_NUM)))

            if total_loss_test < best_loss:
                best_loss = total_loss_test
                print("Saving the model and zero the flag ")

                model_file_name = '%s/ResGAT_model_noMLP_distance.pt' % MODEL_FILE_PATH
                if os.path.exists(model_file_name):
                    os.remove(model_file_name)
                os.system("mkdir -p %s" % (os.path.dirname(model_file_name)))
                torch.save(model_res_distance.state_dict(), model_file_name)
                model_file_name = '%s/ResGAT_model_noMLP_position.pt' % MODEL_FILE_PATH

                if os.path.exists(model_file_name):
                    os.remove(model_file_name)
                os.system("mkdir -p %s" % (os.path.dirname(model_file_name)))
                torch.save(model_res_position.state_dict(), model_file_name)

                print("best loss=", best_loss /
                      (len(self.test_feature) // GROUP_NUM))  # 平均一张图的所有macro损失
                flag = 0
            else:
                flag += 1
                print("flag = %d" % flag)
                if flag == ITERATION:
                    print("best loss=", best_loss /
                          (len(self.test_feature) // GROUP_NUM))  # 平均一张图的所有macro损失
                    print("******************** End of Training ********************")
                    break

            print("************************************************************")

    def load_res_model_noMLP(self):
        self.final_feature[:, [self.my_order]] = self.final_feature[:, [self.random_order]]
        self.final_distance[:, [self.my_order]] = self.final_distance[:, [self.random_order]]
        self.final_adj[:, [self.my_order]] = self.final_adj[:, [self.random_order]]

        model_res_distance = ResGAT_Distance_noMLP(
            len(self.final_feature[0]), int(2), int(len(self.final_feature[0]) / 2))
        model_res_distance.to(DEVICE)
        model_res_distance.load_state_dict(torch.load(
            '%s/ResGAT_model_noMLP_distance.pt' % MODEL_FILE_PATH))
        model_res_distance.eval()

        model_res_position = ResGAT_Position_noMLP(
            len(self.final_feature[0]), int(2), int(len(self.final_feature[0]) / 2), int(2))
        model_res_position.to(DEVICE)
        model_res_position.load_state_dict(torch.load(
            '%s/ResGAT_model_noMLP_position.pt' % MODEL_FILE_PATH))
        model_res_position.eval()

        feature_pos = self.final_feature
        adj_matrix_pos = self.final_adj

        distance_gat_output, prediction_distance = model_res_distance(
            self.final_feature, self.final_adj)
        prediction_position = model_res_position(
            feature_pos, distance_gat_output, adj_matrix_pos)
        prediction_position = prediction_position.cpu()
        prediction_position = prediction_position.detach().numpy()
        self.restore_order_and_save_result(prediction_distance, prediction_position)

    def load_res_model_without_distance_noMLP():
        adj_matrix, feature, my_order, random_order = load_final_data_without_distance()

        feature[:, [my_order]] = feature[:, [random_order]]
        adj_matrix[:, [my_order]] = adj_matrix[:, [random_order]]
        model_res_distance = ResGAT_Distance_noMLP(
            len(feature[0]), int(2), int(len(feature[0]) / 2))
        model_res_distance.to(DEVICE)

        model_res_distance.load_state_dict(torch.load(
            '%s/ResGAT_model_distance.pt' % MODEL_FILE_PATH))
        model_res_distance.eval()

        model_res_position = ResGAT_Position_noMLP(
            len(feature[0]), int(2), int(len(feature[0]) / 2), int(2))
        model_res_position.to(DEVICE)

        model_res_position.load_state_dict(torch.load(
            '%s/ResGAT_model_position.pt' % MODEL_FILE_PATH))
        model_res_position.eval()

        feature_pos = feature
        adj_matrix_pos = adj_matrix
        distance_gat_output, prediction_distance = model_res_distance(
            feature, adj_matrix)
        prediction_position = model_res_position(
            feature_pos, distance_gat_output, adj_matrix_pos)
        prediction_position = prediction_position.cpu()
        prediction_position = prediction_position.detach().numpy()

        self.restore_order_and_save_result_without_distance(
            my_order, prediction_position, random_order)
        print("Ending")


if __name__ == '__main__':
    # two_layer_model
    # train_and_test_distance_model()
    # train_and_test_position_model()
    # load_two_layer_model()

    # ResGAT_Model
    train_res_gat_model()
    load_res_model()
    # load_res_model_without_distance()
    # load_res_model_without_distance_and_shuffle()

    # ResGAT_Model_noMLP
    # train_res_gat_model_noMLP()
    # load_res_model_noMLP()
    # load_res_model_without_distance_noMLP()
