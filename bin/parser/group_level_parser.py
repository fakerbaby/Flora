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
    Extract macros from .pl file and recognoze each one as a sole cluster.
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
    the matrix by means of adding a few auxillary lines and merging sole marginal nodes to increase the density
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
        print("start to extend cluster...")
        macro_list, macro_num = extract_macro(pl_path)
        print("extract macros succeed! There are {macro} macros".format(macro = macro_num))
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
            for j in range(i, self.cluster_number):
                feature[i,j] = feature[i,j] + feature[j,i]
        for i in range(self.cluster_number):
            feature[:, i] = feature[i, :]
        
        self.adjacent_matrix = calculate_adjacent_matrix(feature)
        self.feature_matrix = feature
        print("extend cluster finished.")
            
            
    def establish_result_cluster(self):
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
            update = [0] * len(one_has_adj)  # record the updated cluster
            drop = []                       #record the orphan cluster
            for i in range(len(one_has_adj)):
                if self.cluster_number_record > expect_cluster and update[i] == 0: 
                    if neighbor[i] == one_has_adj[i]: 
                        #only 2-connectivity neighbor clusters -> 1 cluster, then drop it 
                        # update[i] = one_has_adj[i]
                        # drop.append(neighbor[i])
                        # drop.append(one_has_adj[i])
                        # self.cluster_number_record -= 2
                        #todo
                        pass
                    else:
                        update[i] = neighbor[i]
                        drop.append(one_has_adj[i])
                        self.cluster_number_record -= 1
                if self.cluster_number_record == expect_cluster:
                    return update, drop
        
        
        no_adj_record = []
        one_adj_has_record = []
        one_nei_node_record = []
        result_cluster_record = []
        size = len(self.feature_matrix)
        print("start to modify the feature matrix...")
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
        self.cluster_number_record = self.cluster_number - len(no_adj_record)
        # assign expect_cluster number primarily is mainly to represent it as N^2 form, 
        # so that the matix can be a square.
        square = int(np.sqrt(self.cluster_number_record))
        require = 0
        for i in range(square,0,-1):
            if self.cluster_number_record - len(one_adj_has_record) < i**2:
                require = i**2
                break        #one adjacent matrix qualify 
        #update cluster id
        print("start to remove some orphan cluster or only on adj cluster...")
        self.update_cluster, self.drop_cluster = modify_cluster(require, one_adj_has_record, one_nei_node_record)
        self.drop_cluster.extend(no_adj_record)
        
        #update cluster id
        for cluster in self.ori_cluster:
            if cluster in one_adj_has_record and self.update_cluster[one_adj_has_record.index(cluster)] > 0:
                result_cluster_record.append(self.update_cluster[one_adj_has_record.index(cluster)])
            else:
                result_cluster_record.append(cluster)
        self.result_cluster = result_cluster_record
        
        #update adjacent matrix
        self.adjacent_matrix = np.delete(self.adjacent_matrix,self.drop_cluster,0)
        self.adjacent_matrix = np.delete(self.adjacent_matrix,self.drop_cluster,1)
        #update feature matrix
        self.feature_matrix = np.delete(self.feature_matrix,self.drop_cluster,0)
        self.feature_matrix = np.delete(self.feature_matrix,self.drop_cluster,1)
        
        
    def add_conncetivity(self, threshold, weight_1, weight_2):
        """[summary]

        Args:
            threshold ([type]): [description]

        Returns:
            [type]: [description]
        """
        size = len(self.feature_matrix)
        two_step_nei_feature = np.zeros((size,size))  #can record neighbor's neighbor
        print("start to add connectivity to make the feature matrix desnser...")
        for i in range(size):
            two_step_nei = []
            two_step_nei_list = []
            com_neighbor = [] # record common neighbor
            num_1 = np.sum(self.feature_matrix[i]>0)
            num_2 = np.sum(two_step_nei_feature[i]>0)
            adj_num = num_1 + num_2  # total neighbor cells of each cluster
            if adj_num < threshold:
                # neighbor 
                one_step_nei = np.where(self.feature_matrix[i] > 0)     #one neighbor 
                # print(one_step_nei)
                for nei in one_step_nei[0]:
                    # neighbor's neighbor
                    two_step_nei.append(np.where(self.feature_matrix[nei] > 0))   #neighbor's neighbor
                for _nei in range(len(two_step_nei)):
                    two_step_nei_list.append(list(two_step_nei[_nei][0]))  #neighbor's neighbor convert to list([list],[list],...])
                #set initialization
                sum_com_set = set(two_step_nei[0][0])   
                for nei in two_step_nei_list:
                    sum_com_set = sum_com_set.intersection(set(nei))
                    com_neighbor.extend(list(sum_com_set))  # 
                sum_com_set = list(sum_com_set)
                
                com_neighbor = com_neighbor[1:]
                com_neighbor = list(set(com_neighbor)) 
                # others
                if len(sum_com_set) + adj_num >= threshold: 
                    sum_com_set = sum_com_set[0:threshold - adj_num] 
                else:
                    com_neighbor = com_neighbor[0:threshold - adj_num - len(sum_com_set)] 
                
                for x in range(len(sum_com_set)):
                    two_step_nei_feature[i][sum_com_set[x]] = weight_1
                    two_step_nei_feature[sum_com_set[x]][i] = weight_1
                for y in range(len(com_neighbor)):
                    if two_step_nei_feature[i][com_neighbor[y]] == 0:
                        two_step_nei_feature[i][com_neighbor[y]] = weight_2
                        two_step_nei_feature[com_neighbor[y]][i] = weight_2
        self.feature_matrix = self.feature_matrix * 3 + two_step_nei_feature
        print("feature matrix process finished!")
        return self.feature_matrix    
        
        
    def save_data(self, modified_cluster_path, adj_path, feature_path, result_cluster_path):
        """[summary]

        Args:
            modified_cluster_path ([type]): [description]
            adj_path ([type]): [description]
            feature_path ([type]): [description]
        """
        print("saving...")
        #save cluster
        save_extended_cluster(modified_cluster_path, self.ori_cluster)
        #save adjacent matrix
        save_adjacent_matrix(adj_path, self.adjacent_matrix)
        #
        save_feature_matrix(feature_path, self.feature_matrix)
        #
        save_result_matrix(result_cluster_path, self.result_cluster)
        print("saved successfully!")