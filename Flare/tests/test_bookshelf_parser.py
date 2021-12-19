from _path import NET_PATH, NODE_PATH, FIRST_MAT_PATH, SUB_NET_PATH, NODE2MAT_PATH
import bin.parser.bookshelf_parser as parser

net_path, node_path, target_path = NET_PATH, NODE_PATH, FIRST_MAT_PATH 
subnet_path,node2matrix_path = SUB_NET_PATH, NODE2MAT_PATH
test = parser.BookshelfParser()
test.load_data(net_path, node_path)
test.net_to_matrix(target_path, node2matrix_path, subnet_path)
test.monitor()
