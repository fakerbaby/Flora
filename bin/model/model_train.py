import torch
from torch import nn, optim
import random
from model import model1, model2,model2_2GAT,GraphAttentionLayer,multiGAT_2layer,multiGAT_2mlp,GAT_2line
from readdata import read_data
import numpy as np
import math

# # input the wire connection relationship, output the 2d-position
#####todo #######################

def test_model1_device(model, feature, adj, label, N, dis):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    s, final_loss = 0, 0

    for i in range(len(feature) // N):     # determine how many graphs in test data set
        Adj = adj[N * i:(N * (i + 1))][:]
        F = feature[N * i:(N * (i + 1))][:]
        L = label[N * i:(N * (i + 1))][:]
        D = dis[N * i:(N * (i + 1))][:]

        my_or = list(range(0, len(F)))
        random_or = list(range(0, len(F)))
        random.shuffle(random_or)
        # change the row order
        L[my_or] = L[random_or]
        F[my_or] = F[random_or]
        Adj[my_or] = Adj[random_or]
        D[my_or] = D[random_or]
        # change the column order
        F[:, [my_or]] = F[:, [random_or]]
        Adj[:, [my_or]] = Adj[:, [random_or]]
        D[:, [my_or]] = D[:, [random_or]]

        F = F.to(device)
        Adj = Adj.to(device)
        L = L.to(device)
        D = D.to(device)

        pre = model(F, Adj)

        try:
            ave_loss = loss_func(pre, L, Adj, N, D)  # lost on per-chip
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING4: ran out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise e

        s += ave_loss.item()   # accumulate the loss on all chips all nodes
    return s, pre


def distance_calculation(pre, adj, num):
    s = torch.ones(num,num)
    s = s.to(device)
    x0 = s*pre[:,0]
    x1 = x0.T
    X = (x0 - x1)**2
    y0 = s*pre[:,1]
    y1 = y0.T
    Y = (y0 - y1)**2

    pre_dis_array = X+Y

    mask_dis = torch.mul(pre_dis_array, adj)
    # mask_dis = mask_dis.to(device)

    return mask_dis


def overlap_loss2(pre_pos):
    pre_pos = pre_pos.cpu()
    pre_pos = pre_pos+0.5

    pre_pos_int = torch.trunc(pre_pos)
    x = math.sqrt(N)
    add_num = torch.tensor([N-1])
    occupy = pre_pos_int[:, 0] * x + pre_pos_int[:, 1]

    occupy_np = occupy.detach().numpy()
    occupy_np = np.append(occupy_np, [N - 1])
    occupy_np = occupy_np.astype(int)
    # print(type(occupy_np),occupy_np.dtype)

    distri = np.bincount(occupy_np)
    distri = distri.astype(float)
    print(distri)
    distri_tensor = torch.from_numpy(distri[0:N])
    distri_tensor.requires_grad = True

    # overlap = torch.sum(distri_tensor)

    # sort_occupy = torch.sort(occupy, dim=-1, descending=False, out=None).values
    # print(sort_occupy, sort_occupy.dtype)

    # scale_out = np.sum((occupy - N-1) > 0)  # Count the number of elements beyond the boundary
    # occupy_list = occupy.tolist()
    # overlap_scale = N - len(list(set(occupy_list))) + scale_out
    return distri_tensor


def overlap_loss3(pre_pos):
    pre_pos = pre_pos.cpu()
    pre_pos = pre_pos+0.5

    pre_pos_int = torch.trunc(pre_pos)
    x = math.sqrt(N)
    occupy = pre_pos_int[:, 0] * x + pre_pos_int[:, 1]
    occupy_sort = torch.sort(occupy)

    return occupy_sort


def overlap_loss(pre_pos):
    help_matrix = torch.zeros(N,N)
    pre_pos = pre_pos.cpu()
    pre_pos = pre_pos+0.5
    pre_pos_int = torch.trunc(pre_pos)
    x = math.sqrt(N)
    occupy = pre_pos_int[:, 0] * x + pre_pos_int[:, 1]
    occupy = occupy + 1
    occupy_int = occupy.int()
    occupy = occupy.unsqueeze(0)

    for i in range(len(occupy_int)):
        if 0 <= occupy_int[i] <= 899:
            help_matrix[i][occupy_int[i] - 1] = 1 / occupy_int[i]

    distribution = torch.mm(occupy, help_matrix)
    zero = torch.zeros(1, N)
    one = torch.ones(1, N)
    help_matrix = torch.where(distribution > 0, zero, one)
    help_matrix = help_matrix * 100
    distribution = distribution + help_matrix

    # flag_matrix = torch.where(distribution > 0, zero, one)
    # flag_matrix = flag_matrix * 100
    # distribution = distribution + flag_matrix
    distribution = torch.squeeze(distribution, 0)
    print(distribution)
    return distribution


def loss_func(pre_pos, label_pos, adj_matrix, node_num, label_dis):
    loss_function = nn.MSELoss()
    loss1 = loss_function(pre_pos, label_pos)
    loss1 = loss1.cpu()
    print("loss1 :", loss1)  # distance between pre_pos and label_pos

    pre_dis = distance_calculation(pre_pos, adj_matrix, node_num)
    mask_label = torch.mul(label_dis, adj_matrix)
    print(torch.sum(pre_dis))
    print(torch.sum(mask_label))
    loss2 = loss_function(pre_dis, mask_label)
    loss2 = loss2.cpu()
    print("loss2: ", loss2)  # total wirelength

    # label_overlap = torch.ones(N)
    # pre_overlap = overlap_loss(pre_pos)
    #
    # # loss3_mean = torch.sum(pre_overlap) / torch.count_nonzero(pre_overlap)
    # # loss3_var = torch.var(pre_overlap)
    # # pre_occupy = pre_occupy.to(torch.float32)
    # loss3 = loss_function(pre_overlap, label_overlap)
    # loss3 = loss3.cpu()
    # print("loss3: ", loss3)  # overlap loss

    # loss_total = loss1 + loss2 + loss3
    loss_total = loss1 + loss2
    # loss_total = loss2
    print("total_loss: ", loss_total)

    return loss_total


def loadModel():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    compact = 2

    feature = np.loadtxt("5training_data/576/adaptec1/finalfeature.csv", delimiter=",")
    adj_matrix = np.zeros(shape=(len(feature), len(feature[0])), dtype="double")
    connectivity_index = np.where(feature > 0)  # connectivity_index is a tuple data, each element is an array
    for i, j in zip(connectivity_index[0], connectivity_index[1]):
        adj_matrix[i][j] = 1
    for k in range(len(feature[0])):
        adj_matrix[k][k] = 1

    feature = torch.from_numpy(feature).float().to(device)
    adj_matrix = torch.from_numpy(adj_matrix).float().to(device)

    # load model
    # model = GraphAttentionLayer(len(feature[0]), int(compact), 0.2, 0.2).to(device)
    model = GAT_2line(len(feature[0]), int(compact), 0.2, 0.2).to(device)

    model.load_state_dict(torch.load('6model/save_model/GAT2layer_576_0928_addn.pt'))
    model.eval()

    my_order = list(range(0, len(feature)))
    random_order = list(range(0, len(feature)))
    random.shuffle(random_order)
    # change the row order
    feature[my_order] = feature[random_order]
    adj_matrix[my_order] = adj_matrix[random_order]
    # label[my_order] = label[random_order]

    feature[:, [my_order]] = feature[:, [random_order]]
    adj_matrix[:, [my_order]] = adj_matrix[:, [random_order]]

    prediction = model(feature, adj_matrix)
    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()

    final_order = []
    for i in range(len(random_order)):
        final_order.append(random_order.index(i))
    prediction[my_order] = prediction[final_order]

    np.savetxt('6model/result/GAT2layer_576_0928_addn.csv', prediction, delimiter=',')


def lodaModel_real_benchmark(b_name):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    compact = 2

    feature = np.loadtxt('4-1cluster_feature/' + b_name + '/add_neighbor/feature576.csv',
                         delimiter=",")

    adj_matrix = np.zeros(shape=(len(feature), len(feature[0])), dtype="float")
    connectivity_index = np.where(feature > 0)  # connectivity_index is a tuple data, each element is an array
    for i, j in zip(connectivity_index[0], connectivity_index[1]):
        adj_matrix[i][j] = 1
    for k in range(len(feature[0])):
        adj_matrix[k][k] = 1

    feature = torch.from_numpy(feature).float().to(device)
    adj_matrix = torch.from_numpy(adj_matrix).float().to(device)

    # load model
    model = GAT_2line(len(feature[0]), int(compact), 0.2, 0.2).to(device)
    # model = GraphAttentionLayer(len(feature[0]), int(compact), 0.2, 0.2).to(device)
    model.load_state_dict(torch.load('6model/save_model/GAT2layer_576_0928_addn.pt'))
    model.eval()

    prediction = model(feature, adj_matrix)
    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()

    np.savetxt('6model/result/'+b_name+'_0928_GAT2layer_addn.csv', prediction, delimiter=',')


if __name__ == '__main__':
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # !NEED CHANGE
    compact = 2  # the dimension of GAT compacted feature
    N = 576     # the node number
    # !NEED CHANGE
    testfeature, testadj, testlabel = read_data('5training_data/576/adaptec1/testlabel.csv')
    testdistance = np.loadtxt('5training_data/576/adaptec1/testdistance.csv', delimiter=',')
    print("testdata ready")
    trainfeature, trainadj, trainlabel = read_data('5training_data/576/adaptec1/trainlabel.csv')
    traindistance = np.loadtxt('5training_data/576/adaptec1/traindistance.csv', delimiter=',')
    print("traindata ready")

    # !NEED CHANGE
    # Pos_model = GraphAttentionLayer(len(trainfeature[0]), int(compact), 0.2, 0.2).to(device)
    Pos_model = GAT_2line(len(trainfeature[0]), int(compact), 0.2, 0.2).to(device)

    optimizer = optim.Adam(Pos_model.parameters(), lr=1e-4)
    max_iterations = 500
    adjacency_matrix = []
    final_adjacency_matrix = []
    train_pic = []
    test_pic = []
    best_point = []

    flag = 0
    best_loss = float("inf")
    lost, best_lost_ave_den, best_lost_connect_den, best_lost_without_connect_den = [], [], [], []
    for iteration in range(max_iterations):
        s = 0
        for i in range(len(trainfeature) // N):  # set batch  per graph per time
            trainL = trainlabel[N * i:(N * (i + 1))][:]
            trainAdj = trainadj[N * i:(N * (i + 1))][:]
            trainF = trainfeature[N * i:(N * (i + 1))][:]
            trainD = traindistance[N * i:(N * (i + 1))][:]

            my_order = list(range(0, len(trainF)))
            random_order = list(range(0, len(trainF)))
            random.shuffle(random_order)
            # change the row order
            trainL[my_order] = trainL[random_order]
            trainF[my_order] = trainF[random_order]
            trainAdj[my_order] = trainAdj[random_order]
            trainD[my_order] = trainD[random_order]

            # change the column order
            trainF[:, [my_order]] = trainF[:, [random_order]]
            trainAdj[:, [my_order]] = trainAdj[:, [random_order]]
            trainD[:, [my_order]] = trainD[:, [random_order]]

            # covert numpy.array to tensor
            trainL = torch.from_numpy(trainL).float()
            trainF = torch.from_numpy(trainF).float()
            trainAdj = torch.from_numpy(trainAdj).float()
            trainD = torch.from_numpy(trainD).float()

            trainF = trainF.to(device)
            trainAdj = trainAdj.to(device)
            trainL = trainL.to(device)
            trainD = trainD.to(device)

            # training the model

            lytPre = Pos_model(trainF, trainAdj)

            try:
                loss = loss_func(lytPre, trainL, trainAdj, N, trainD)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING2: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e

            s += loss.item()  # add all graph loss together in one epoch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Train Result:", s / (len(trainfeature) // N), "\titeration=", iteration)
        train_pic.append(s / (len(trainfeature) // N))

        # testing the model
        testadj = torch.Tensor(testadj)  # convert numpy.array to Tensor
        testfeature = torch.Tensor(testfeature)
        testlabel = torch.Tensor(testlabel)
        testdistance = torch.Tensor(testdistance)
        total_loss_test, embedding = test_model1_device(Pos_model, testfeature, testadj, testlabel, N, testdistance)
        # test_pic.append(total_loss_test / (len(testfeature) // N))

        print("Test Result:", total_loss_test / (len(testfeature) // N), "\titeration=", iteration)  # 平均一张图的所有macro损失
        if total_loss_test < best_loss:
            best_loss = total_loss_test
            torch.save(Pos_model.state_dict(), '6model/save_model/GAT2layer_576_0928_addn.pt')
            print("best loss=", best_loss / (len(testfeature) // N))  # 平均一张图的所有macro损失
            best_point.append(iteration)
            best_point.append(best_loss / (len(testfeature) // N))
            # adjacency_matrix.append(embedding)  # 保存test里生成的adjacency matrix
            flag = 0
        else:
            flag += 1
            if flag == 10:
                break
    # # # train_test_pic(train_pic, test_pic, best_point)
    print("load model")
    loadModel()
    benchmark_name = 'adaptec1'
    lodaModel_real_benchmark(benchmark_name)
    print("Done")

