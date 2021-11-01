# Author: shenwei
# Date: 2021.10.21
# 格式必须与原本的bookshelf一样，否则无法正确识别内容
# This script is a parser to make the real benchmark used in DreamPlace
# to be compatible with our spectral_cluster program or GAT program
# ***     NOTE: the input of our spectral_cluster is a matrix containing connectivity info      ***
# -------------------------------------------------------------------------------------------
# Output matrix structure should be like:
#         |   0   |  1  |  2   |  3   | 4  | ... | ... |9999 | 10000 |
#     0   |   0   |  25 |  20  |  16  | 0  | ... | ... |  0  |   0   |
#     1   |   25  |  0  |  7   |  0   | 0  | ... | ... |  0  |   0   |
#     2   |   20  |  7  |  0   |  1   | 0  | ... | ... |  0  |   0   |
#     3   |   16  |  0  |  1   |  0   | 0  | ... | ... |  0  |   0   |
#     4   |   0   |  0  |  0   |  0   | 0  | ... | ... |  0  |   0   |
#    . .  |   .   |     |      |      |    | ... | ... |     |       |
#    . .  |   .   |     |      |      |    | ... | ... |     |       |
#    . .  |   .   |     |      |      |    | ... | ... |     |       |
#    9999 |   0   |  0  |   0  |  0   | 0  | ... | ... |  0  |   21  |
#   10000 |   0   |  0  |   0  |  0   | 0  | ... | ... |  21 |   0   |
# -------------------------------------------------------------------------------------------
# Each element in above matrix represents the number of routes between corresponding cells
# There are total 10000 cells in above example.
# ***                               How to implement                                    ***
# -------------------------------------------------------------------------------------------
# The input file is .nets file which contains all the info for each net
# We need create a connectivity matrix based on all nets info
# In each net, the first node can be regarded as the core node, and the other nodes in the
# same net are only connected with core node(the first node)

# Sparse Matrix Construction

# ============================================= REFERENCE ==============================================
# Sparse Matrix Types:
# There are six types of sparse matrices implemented under scipy:
# For more info see: (http://docs.scipy.org/doc/scipy/reference/sparse.html)
#       bsr_matrix -- block sparse row matrix
#       coo_matrix -- sparse matrix in coordinate format
#       csc_matrix -- compressed sparse column matrix
#       csr_matrix -- compressed sparse row matrix
#       dia_matrix -- sparse matrix with diagonal storage
#       dok_matrix -- dictionary of keys based sparse matrix
#       lil_matrix -- row-based linked list sparse matrix
# When to use which matrix:
# The following are scenarios when you would want to choose one sparse matrix type over the another:
#       Fast Arithmetic Operation: cscmatrix, csrmatrix
#       Fast Column Slicing (e.g., A[:, 1:2]): csc_matrix
#       Fast Row Slicing (e.g., A[1:2, :]) csr_matrix
#       Fast Matrix vector products: csrmatrix, bsrmatrix, csc_matrix
#       Fast Changing of sparsity (e.g., adding entries to matrix): lilmatrix, dokmatrix
#       Fast conversion to other sparse formats: coo_matrix
#       Constructing Large Sparse Matrices: coo_matrix


import os
import re
from pathlib import Path
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, save_npz, load_npz

# load benchmark and parser to a 'bookshelf' file
# read .net dataset and capture the key information.

# benchmark_name = ['ispd2005', 'ispd2014', 'ispd2015', 'ispd2019']
# dataset = ['adaptec1', 'adaptec2', 'adaptec3', 'adaptec4',
#            'bigblue1', 'bigblue2', 'bigblue3', 'bigblue4']
# net_path = os.path.join(os.getcwd(), 'benchmark',
#                         benchmark_name[0], dataset[0]+'.nets')
# node_path = os.path.join(os.getcwd(), 'benchmark',
#                          benchmark_name[0], dataset[0]+'.nodes')
# target_path = os.path.join(os.getcwd(), 'benchmark',
#                            benchmark_name[0], 'bookshelf', dataset[0]+'.npz')


class BookshelfParser:
    """
    This Module is mainly to parser Bookshelf into a sparse matrix
    """
    def __init__(self):
        print("===============bookshelf_parser=================")
        self.data = {} 

    def load_data(self, dataset):
        """"
        
        """"
        print("load the benchmarks...")
        # load .node file and .net file
        with open(net_path, 'r') as f:
            benchmark = f.read()
        with open(node_path, 'r') as f:
            node_benchmark = f.read()
        
        
    def capture_key_parameter(self):
        """ read info from .net file and .node file
        store the net file path & the node file path of benchmark
        use dictionary data to store the key parameter from benchmark
        :param: None
        :return:
            "param row": a Integer list carrying the core_node
            "param column": a Integer list carrying the adjcent_node
        """
        

        # capture the parameter from the benchmark files
        node_num_index = node_benchmark.find('NumNodes')
        node_teminal_index = node_benchmark.find('NumTerminals')
        node_num = node_benchmark[node_benchmark.find(
            ':', node_num_index)+1: node_teminal_index]
        node_teminal = int(node_benchmark[node_benchmark.find(
            ':', node_teminal_index)+1: node_benchmark.find('\n', node_teminal_index)])
        if node_num.isdigit():
            node_num = int(node_num)
        node_num = int(node_num[:-1])
        self.data['NumNodes'] = node_num
        self.data['NumTerminals'] = node_teminal
        print("=========start to capture the key parameters========")
        net_num_index = benchmark.find('NumNets')
        net_pins_index = benchmark.find('NumPins')
        num_nets = int(benchmark[benchmark.find(
            ':', net_num_index) + 1: net_pins_index])
        num_pins = int(benchmark[benchmark.find(
            ':', net_pins_index) + 1: benchmark.find('\n', net_pins_index)])

        self.data['NumNets'] = num_nets
        self.data['NumPins'] = num_pins
        row, column = [], []
        net_degree_index = benchmark.find('NetDegree')
        while True:
            num_subnet_degrees = int(
                benchmark[benchmark.find(':', net_degree_index) + 1: benchmark.find('n', net_degree_index)])
            for sub_net_index in range(num_subnet_degrees):
                if sub_net_index == 0:
                    start_index = benchmark.find('\n', net_degree_index)
                    end_index = benchmark.find('\n', start_index + 1)
                    core_node_str = benchmark[start_index: end_index]
                    # regex pattern to search
                    pattern = re.compile(r'\d+\b')
                    search = pattern.search(core_node_str)
                    core_node = search.group(0)
                else:
                    start_index = benchmark.find('\n', end_index)
                    end_index = benchmark.find('\n', start_index + 1)
                    adjacent_node_str = benchmark[start_index: end_index]
                    pattern = re.compile(r'\d+\b')
                    search = pattern.search(adjacent_node_str)
                    adjacent_node = search.group(0)
                    column.append(int(adjacent_node))

            for i in range(num_subnet_degrees - 1):
                row.append(int(core_node))
            net_degree_index = benchmark.find(
                'NetDegree', net_degree_index + 1)
            if net_degree_index == -1:
                break
        print("capture key paramters success!")
        return row, column

    def convert_to_matrix(self, row, column):
        """ convert the benchmark to an sparse adjcent Matrix
        "param row": a Integer list carrying the core_node
        "param column": a Integer list carrying the adjcent_node
        :return: sparse_connectivity_matrix: each element represents the number of routes for corresponding cells
        """
        # sparse the matrix
        # lil_matrix Fast Changing of Sparsity(eg. , adding entries to matrix)
        num_nodes = self.data['NumNodes']
        sparse_connectivity_matrix = lil_matrix((num_nodes, num_nodes))
        for (i, j) in zip(row, column):
            if i == j:
                continue
            sparse_connectivity_matrix[i, j] += 1
            sparse_connectivity_matrix[j, i] += 1
        # csc_matrix can faster arithmetics operations
        print("start convert benchmark files into sparse matrix...")
        sparse_connectivity_matrix = csc_matrix(sparse_connectivity_matrix)
        print("convert success!")
        return sparse_connectivity_matrix

    def save_to_(self, data, target_path):
        """ save to target files
        "param data": sparse_connectivity_matrix
        "param target_path": the path to taget file
        """
        print("==========saving=============")
        print("start saving files, please wait for a few seconds......")
        abspath = os.path.abspath(target_path)
        print("data will be saved to", abspath)
        save_npz(target_path, data)
        print("data has been saved!")
        print("=============bookshelf_parser===============")


if __name__ == '__main__':
    test = bookshelf_parser()
    row, col = test.capture_key_parameter()
    data = test.convert_to_matrix(row, col)
    test.save_to_(data, target_path)
