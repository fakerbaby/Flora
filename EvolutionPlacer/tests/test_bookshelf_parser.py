import sys
from _const import BASE_DIR,BENCHMARK,DATASET
from test import generate_path, generate_json_path
sys.path.append(BASE_DIR)
import bin.parser.bookshelf_parser as parser



net_path, node_path, target_path = generate_path(BENCHMARK,DATASET)
json_path = generate_json_path(BENCHMARK,DATASET)
x = parser.BookshelfParser()
x.load_data(net_path, node_path)
x.monitor()
x.net_to_matrix(target_path, json_path)