import os
import json
import numpy as np
from scipy.sparse import lil_matrix


def establish_adjacent_matrix(feature):
    """
    Calculate the adjacent matrix induced from the feature
    Args:
        feature: a feature matrix
    Returns: 
        adjacent: a matrix of the adjacent matrix produced by the feature
    """
    dimension = len(feature)
    adjacent = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            adjacent[i, j] = 1 if feature[i, j] > 0 or i == j else None
    return adjacent


def extract_macro(pl_path):
    """
    Extract macros from .pl file and recognoze each one as a solo cluster.
    And then need to calculate the adjacent relationship with their neighbors. 
    Args:
        pl_path: the path to the .pl files
    Returns:
        macro_list: 
        macro_num:
    """
    macro_list = []
    with open(pl_path, mode='r') as f:
        for line in f.readlines():
            if line.find('FIXED') != -1:
                macro_list.append(line.split()[0])
    macro_num = len(macro_list)
    return macro_list, macro_num


def load_net_list(net_list_path):
    """
    Returns:
        net_list: a dict
    """
    with open(net_list_path, 'r') as f:
        net_list = json.load(f)
    return net_list


def load_ori_cluster(ori_cluster_path):
    """

    Returns:
       ori_cluster: a list 
    """
    with open(ori_cluster_path, 'r') as f:
        ori_cluster = np.loadtxt(f, dtype=int, delimiter=',')
    return ori_cluster


def save_modefied_cluster(path, new_cluster):
    """
    """
    np.savetxt(path, new_cluster, delimiter=',')  # save cluster


class GroupLevelParser:
    """
    Parser for Group Level
    This parser module is mainly to parser the cluster level information into a new matrix, and we extract fixed cells(macro)
    from the cluster level information and recognize each of them as one cluster, which is the counterpart as the clusters produced by 
    spectral clustering algorithm. Considering the over-sparse matrix can not make the GAT model learn very well, so we need to modify
    the matrix by means of adding a few auxillary lines and merging solo marginal nodes to increase the density
    of the matrix. 
    """

    def __init__(self):
        self.cluster_number = 0
        self.ori_cluster = None
        self.net_list = None

    def load_data(self, net_list_path, ori_cluster_path):
        """
        Args:

        Returns:

        """
        net_list = load_net_list(net_list_path)
        ori_cluster = load_ori_cluster(ori_cluster_path)
        self.ori_cluster, self.net_list = ori_cluster, net_list

    def modify_cluster(self, group_number, pl_path, modified_cluster_path):
        """
        This function is the major work of the GroupLevelParser do
        Args:
            ori_cluster: the origin result file of the spectral clustering
            net_list: a list which composed of the subnet of nodes
            pl_path: the path to the .pl file
            group_number: the number of origin spectral clustering
        Returns:

        """
        macro_list, macro_num = extract_macro(pl_path)
        # update the number of cluster
        self.cluster_number = macro_num + group_number
        # generate a empty matrix
        feature = np.zeros((self.cluster_number, self.cluster_number))

        for macro in enumerate(macro_list):
            self.ori_cluster[int(macro[1][1:])] = group_number + macro[0]
        for core_cell, adjacent_cell_list in self.net_list.items():
            for adjacent_cell in adjacent_cell_list:
                if self.ori_cluster[int(core_cell)] != self.ori_cluster[adjacent_cell]:
                    start_point = self.ori_cluster[int(core_cell)]
                    end_point = self.ori_cluster[adjacent_cell]
                    feature[start_point, end_point] += 1

        for i in range(self.cluster_number):
            for j in range(self.cluster_number):
                feature[i][j] = feature[i][j] + feature[j][i]

        for i in range(self.cluster_number):
            feature[i, :] = feature[:, i]
        save_modefied_cluster(modified_cluster_path, self.ori_cluster)
        return feature

    def reduce_margin_node():
        """
        todo
        """
