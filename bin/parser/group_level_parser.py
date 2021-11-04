import os
import json
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix


def calculate_adjacent_matrix(feature):
    """
    Calculate the adjacent matrix induced from the feature
    Args:
        feature ([]): a feature matrix
        
    Returns: 
        [type]: a matrix of the adjacent matrix produced by the feature
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
        pl_path ([]): the path to the .pl files
        
    Returns:
        [type]: [description] 
        [type]: [description]
    """
    macro_list = []
    with open(pl_path, mode='r') as f:
        for line in f.readlines():
            if line.find('FIXED') != -1:
                macro_list.append(line.split()[0])
    macro_num = len(macro_list)
    return macro_list, macro_num


def load_net_list(net_list_path):
    """[summary]

    Args:
        net_list_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    with open(net_list_path, 'r') as f:
        net_list = json.load(f)
    return net_list


def load_ori_cluster(ori_cluster_path):
    """[summary]

    Args:
        ori_cluster_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    with open(ori_cluster_path, 'r') as f:
        ori_cluster = np.loadtxt(f, dtype=int, delimiter=',')
    return ori_cluster


def save_modefied_cluster(modified_cluster_path, new_cluster):
    """[summary]

    Args:
        modified_cluster_path ([type]): [description]
        new_cluster ([type]): [description]
    """
    if not os.path.exists(os.path.dirname(modified_cluster_path)):
        os.makedirs(os.path.dirname(modified_cluster_path))
    np.savetxt(modified_cluster_path, new_cluster, delimiter=',')  # save cluster


def save_adjacent_matrix(adjacent_matrix_path, adjacent_matrix):
    """[summary]

    Args:
        adjacent_matrix_path ([type]): [description]
        adjacent_matrix ([type]): [description]
    """
    if not os.path.exists(os.path.dirname(adjacent_matrix_path)):
        os.makedirs(os.path.dirname(adjacent_matrix_path))
    np.savetxt(adjacent_matrix_path, adjacent_matrix, delimiter=',')
        

def save_feature_matrix(feature_matrix_path, feature_matrix):
    """[summary]

    Args:
        feature_matrix_path ([type]): [description]
        feature_matrix ([type]): [description]
    """
    # with open(feature_matrix_path, 'w') as f:
    if not os.path.exists(os.path.dirname(feature_matrix_path)):
        os.makedirs(os.path.dirname(feature_matrix_path))
    np.savetxt(feature_matrix_path, feature_matrix)
    
class GroupLevelParser:
    """
    [  Parser for Group Level ]
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
        self.adjacent_matrix = None
        self.feature_matrix = None

    def load_data(self, net_list_path, ori_cluster_path):
        """[summary]

        Args:
            net_list_path ([type]): [description]
            ori_cluster_path ([type]): [description]
        """
        net_list = load_net_list(net_list_path)
        ori_cluster = load_ori_cluster(ori_cluster_path)
        self.ori_cluster, self.net_list = ori_cluster, net_list

    def extend_cluster(self, pl_path, group_number):
        """
        This function is one of the GroupLevelParser's major work   
        Args:
            ori_cluster ([]): the origin result file of the spectral clustering
            net_list ([]]): a list which composed of the subnet of nodes
            pl_path ([]): the path to the .pl file
            group_number ([]): the number of origin spectral clustering
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
        
        self.adjacent_matrix = calculate_adjacent_matrix(feature)
        self.feature_matrix = feature

    def save_data(self, modified_cluster_path, adj_path, feature_path):
        """[summary]

        Args:
            modified_cluster_path ([type]): [description]
            adj_path ([type]): [description]
            feature_path ([type]): [description]
        """
        #save cluster
        save_modefied_cluster(modified_cluster_path, self.ori_cluster)
        #save adjacent matrix
        save_adjacent_matrix(adj_path, self.adjacent_matrix)
        #
        save_feature_matrix(feature_path, self.feature_matrix)
        
    
    def modify_cluster(self):
       #todo
        
       
