from _const import BENCHMARK, DATASET
from _path import generate_path, generate_json_path
import bin.parser.bookshelf_parser as parser

net_path, node_path, target_path = generate_path(BENCHMARK,DATASET)
subnet_path,node2matrix_path = generate_json_path(BENCHMARK,DATASET)
test = parser.BookshelfParser()
test.load_data(net_path, node_path)
test.net_to_matrix(target_path, node2matrix_path, subnet_path)
test.monitor()
