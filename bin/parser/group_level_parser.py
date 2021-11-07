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


def save_extended_cluster(extended_cluster_path, new_cluster):
    """[summary]

    Args:
        extended_cluster_path ([type]): [description]
        new_cluster ([type]): [description]
    """
    if not os.path.exists(os.path.dirname(extended_cluster_path)):
        os.makedirs(os.path.dirname(extended_cluster_path))
    np.savetxt(extended_cluster_path, new_cluster, delimiter=',')  # save cluster


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
    np.savetxt(feature_matrix_path, feature_matrix, delimiter=',')

def save_result_matrix(result_result_path, result_result):
    """[summary]

    Args:
        result_result_path ([type]): [description]
        result_result ([type]): [description]

    Returns:
        [type]: [description]
    """
    if not os.path.exists(os.path.dirname(result_result_path)):
        os.makedirs(os.path.dirname(result_result_path))
    np.savetxt(result_result_path, result_result, delimiter=',')
    
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
        self.expect_cluster_number = 0
        self.cluster_number_record = 0
        self.update_cluster = None
        self.drop_cluster = None
        self.result_cluster = None
        # self.no_adj_record = Noen
        # self.one_adj_record = None
    

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
        extract macros from the .pl file, recognize each of them as a new cluster, and add to 
        origin cluster matrix to generate a new extended matrix, and we can calculate the adjacent
        matrix at the same time.
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
        
 
    def add_conncetivity(self):
        size = len(self.feature_matrix)
        output_feature = np.zeros((size,size)) 
        
        one_step_con = []
        two_step_con = []
        com_adj = []
        
        for i in range(size):
            num_1 = np.sum(self.feature_matrix[i]>0)
            num_2 = np.sum(output_feature[i]>0)
            adj_num = num_1 + num_2
            
            if adj_num < self.expect_cluster_number:
                # neighbor 
                one_step_con = np.where(self.feature_matrix[i] > 0) 
                for nei in one_step_con[0]:
                    # neighbor's neighbor
                    two_step_con.append(np.where(self.feature_matrix[i] > 0)) 
        
        # print(two_step_con)
                  
    
    def establish_final_cluster(self):
        """[summary]

        Args:
            expect_cluster ([type]): [description]
        """
        def modify_cluster(expect_cluster, one_has_adj, neighbor):
            """[summary]

            Args:
                expect_cluster ([type]): [description]
                one_has_adj ([type]): [description]
                one_nei ([type]): [description]

            Returns:
                [type]: [description]
            """
            update = [0] * len(one_has_adj)  # 记录更新后的cluster
            drop = []
            for i in range(len(one_has_adj)):
                if self.cluster_number_record > expect_cluster and update[i] == 0:
                    if neighbor[i] == one_has_adj[i]:
                        #2 neighbor clusters -> 1 cluster    
                        update[i] = one_has_adj[i]
                        drop.append(neighbor[i])
                        drop.append(one_has_adj[i])
                        self.cluster_number_record -= 2
                    else:
                        update[i] = neighbor[i]
                        drop.append(neighbor[i])
                        self.cluster_number_record -= 1
                if self.cluster_number_record == expect_cluster:
                    return update, drop
        
        
        no_adj_record = []
        one_adj_has_record = []
        one_nei_node_record = []
        result_cluster_record = []
        size = len(self.feature_matrix)
        #find one-adj & zero-adj group cluster
        for row in range(size):
            tmp_list = []
            for col in range(size):
                if self.feature_matrix[row][col] > 0:
                    tmp_list.append(col)
            if len(tmp_list) == 0:
                no_adj_record.append(row)
            if len(tmp_list) == 1:
                one_adj_has_record.append(row)
                one_nei_node_record.extend(tmp_list)  
        # print("1",no_adj_record)
        # print('2',one_adj_has_record)
        # print('3',one_nei_node_record)
        # self.one_adj_record = one_adj_record
        # self.no_adj_record = no_adj_record
        self.cluster_number_record = self.cluster_number - len(no_adj_record)
        # assign expect_cluster number primarily is mainly to represent it as N^2 form, 
        # so that the matix can be a square.
        square = int(np.sqrt(self.cluster_number_record))**2
        self.expect_cluster_number = square
        #one adjacent matrix qualify 
        #update cluster id
        self.update_cluster, self.drop_cluster = modify_cluster(square, one_adj_has_record, one_nei_node_record)
        print(self.update_cluster)
        self.drop_cluster.extend(no_adj_record)
        print(self.drop_cluster)
        
        #update cluster id
        for cluster in self.ori_cluster:
            if cluster in one_adj_has_record and self.update_cluster[one_adj_has_record.index(cluster)] > 0:
                result_cluster_record.append(self.update_cluster[one_adj_has_record.index(cluster)])
            else:
                result_cluster_record.append(cluster)
        self.result_cluster = result_cluster_record
        print(len(result_cluster_record))
        
        #update adjacent matrix
        self.adjacent_matrix = np.delete(self.adjacent_matrix,self.drop_cluster,0)
        self.adjacent_matrix = np.delete(self.adjacent_matrix,self.drop_cluster,1)
        #update feature matrix
        self.feature_matrix = np.delete(self.feature_matrix,self.drop_cluster,0)
        self.feature_matrix = np.delete(self.feature_matrix,self.drop_cluster,1)
        print(len(self.feature_matrix),len(self.feature_matrix[1]))
        
        
        
    def save_data(self, modified_cluster_path, adj_path, feature_path, result_cluster_path):
        """[summary]

        Args:
            modified_cluster_path ([type]): [description]
            adj_path ([type]): [description]
            feature_path ([type]): [description]
        """
        #save cluster
        save_extended_cluster(modified_cluster_path, self.ori_cluster)
        #save adjacent matrix
        save_adjacent_matrix(adj_path, self.adjacent_matrix)
        #
        save_feature_matrix(feature_path, self.feature_matrix)
        #
        save_result_matrix(result_cluster_path, self.result_cluster)