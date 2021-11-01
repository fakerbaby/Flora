import torch
from torch import nn, optim
import random
from model import GraphAttentionLayer
from readdata import read_data
from showresult import result_pic,train_test_pic
import numpy as np


# # input the wire connection relationship, output the 2d-position



def loadModel():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    feature = np.loadtxt('groupConnectivity1.csv', delimiter=",")
    adj_matrix = np.zeros(shape=(len(feature), len(feature[0])), dtype="double")
    connectivity_index = np.where(feature > 0)  # connectivity_index is a tuple data, each element is an array
    for i, j in zip(connectivity_index[0], connectivity_index[1]):
        adj_matrix[i][j] = 1

    feature = torch.from_numpy(feature).float().to(device)
    adj_matrix = torch.from_numpy(adj_matrix).float().to(device)

    # load model
    model = GraphAttentionLayer(len(feature[0]), int(compact), 0.2, 0.2).to(device)
    model.load_state_dict(torch.load('GAT.pt'))
    model.eval()

    prediction = model(feature, adj_matrix)
    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()

    np.savetxt('GAT.csv', prediction, delimiter=',')



if __name__ == '__main__':

    # !NEED CHANGE
    compact = 2  # the dimension of GAT compacted feature
    N = 900     # the node number

    print("load model")
    loadModel()
    print("Done")

