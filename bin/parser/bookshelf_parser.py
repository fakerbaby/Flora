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


import sys
import re
import json
from pathlib import Path
from scipy.sparse import csc_matrix, lil_matrix, save_npz

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


def load_data(path):
    """
    a basic function to load data
    """
    with open(path, 'r') as f:
        data = f.read()
    return data


def save_dict_as_json(dictionary: dict, savepath: Path):
    """ save dictionary as json file
    Args: 
        dictionary : the source
        savepath: the directory of target file
    Returns: 
        None
    Raise: 
        IOError: An error occurred accessing the savepath
    """
    # create path if not exist
    savepath.parent.mkdir(parents=True, exist_ok=True)
    json_str = json.dumps(dictionary)
    with open(savepath, 'w') as f:
        f.write(json_str)


def extract_nodes_info(nodefilepath: Path) -> list:
    """
    extract all nodes' name as a list
    Args:
        nodefilepath: the directory of node file
    Returns:
        node_info: a list consists name of all nodes
        node_num:
        term_num:
    """
    with open(nodefilepath, 'r')as f:
        info = f.read()
        node_number_regex = re.compile(r'NumNodes\s*:\s*(\d+)\s+')
        terminal_number_regex = re.compile(r'NumTerminals\s*:\s*(\d+)\s+')
        node_num = int(re.findall(node_number_regex, info)[0])
        term_num = int(re.findall(terminal_number_regex, info)[0])
        node_regex = re.compile(r'\s+(o\d+)\s+\d+\s+\d+')
        node_info = re.findall(node_regex, info)
    return node_info, node_num, term_num


def extract_net_info(net_file_path: Path) -> list:
    """
    extract net information as list where each element is a subnet
    Args:
        net_file_path: the directory of net file
    Returns:
        subnet: subnets info list, each element is a string of subnets info
        net_num:
        pin_num:
    """
    with open(net_file_path, 'r') as f:
        netlist = f.read()
        net_num = int(re.findall(r'NumNets\s*:\s*(\d+)', netlist)[0])
        pin_num = int(re.findall(r'NumPins\s*:\s*(\d+)', netlist)[0])
        subnet_regex = re.compile(r'NetDegree\s*:')
        subnet = re.split(subnet_regex, netlist)
        subnet.pop(0)
    return subnet, net_num, pin_num


def extract_nodes_in_subnets(subnets: list) -> tuple:
    """ extract root cell name and adjacent cell name correlating with root cell in each subnet
    Args:
        subnets: a list of subnets in which each element is a string including all info of subnet
    Returns:
        all_root_cell: list of root cell name
        all_adjacent_cell: list of adjacent cell name
        total_pin: total_pins
    """
    
    print("start extracting nodes in subnets")
    all_root_cell = []
    all_adjacent_cell = []
    adjacent_cell_regex = re.compile(r'o\d+')
    total_pin = 0
    pin_regex = re.compile(r'\s*(\d+)\s+n')
    for subnet_info in subnets:
        total_pin += int(re.findall(pin_regex, subnet_info)[0])
        connected_cell = re.findall(adjacent_cell_regex, subnet_info)
        root_cell = connected_cell[0]
        adjacent_cell = connected_cell[1:]
        all_root_cell.append(root_cell)
        all_adjacent_cell.append(adjacent_cell)
    return all_root_cell, all_adjacent_cell, total_pin


class BookshelfParser:
    """
    This module is mainly to exract some basic information of netlist 
    from .bookshelf file, and map them into a sparse matrix.

    Attributes: 
        net_path: 
        node_path:
    """

    def __init__(self):
        print("===============bookshelf_parser=================")
        self.node2matrix_mapper = dict()
        self.node_info = None
        self.net_info = None
        self.term_num = 0
        self.node_num = 0
        self.net_num = 0
        self.pin_num = 0

    def load_data(self, net_path: Path, node_path: Path):
        """
        this part is to load data from .bookshelf file
        Args:
            net_path: path to .net file
            node_path: path to .node file
        Returns: 
            net_bm: .net data
            node_bm: .node data
        Rasie:
            IOError: An error occurred accessing the net_path & node_path
        """
        # load .node file and .net file
        print("loading benchmarks...")
        self.net_info, self.net_num, self.pin_num = extract_net_info(net_path)
        self.node_info, self.net_num, self.term_num = extract_nodes_info(
node_path)

    def net_to_matrix(self, sparseMatrixFile: Path, cellName2MatrixIndex: Path):
        """ convert the benchmark to an sparse adjcent Matrix
        Args:
            nodefilepath:
            netfilepath:
            sparseMatrixFile:
            cellName2MatrixIndex:
        Returns: 
            sparse_connectivity_matrix: each element represents the number of routes for corresponding cells
        """
        # construct node2matrix_mapper dictionary
        for i in enumerate(self.node_info):
            self.node2matrix_mapper.setdefault(self.node_info[i], i)
        if len(self.node2matrix_mapper) == self.node_num:
            print("cell number = matrix index it's ok to save cellName2MatrixIndex")
            save_dict_as_json(self.node2matrix_mapper, cellName2MatrixIndex)
        else:
            print("cell number not equal with matrix index, check the nodes file")
            sys.exit()

        # initialize sparse matrix with all elements 0
        sparse_matrix = lil_matrix((self.node_num, self.node_num))

        # extract connected cells in each subnet
        root, adjacent, total_pin = extract_nodes_in_subnets(self.net_info)

        # get final sparse matrix
        for x, adj in zip(root, adjacent):
            for y in adj:
                sparse_matrix[self.node2matrix_mapper[x],
                              self.node2matrix_mapper[y]] += 1
                sparse_matrix[self.node2matrix_mapper[y],
                              self.node2matrix_mapper[x]] += 1
        if total_pin != self.pin_num:
            print(
                f"all connected cell number is not {self.pin_num}, can't save sparse matrix right now.")
            sys.exit()
        else:
            print("NumPins is good,start saving sparse matrix")
            # save sparse matrix
            sparse_matrix = csc_matrix(sparse_matrix)
            save_npz(sparseMatrixFile, sparse_matrix)
