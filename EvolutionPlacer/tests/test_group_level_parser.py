from _const import BENCHMARK, DATASET, GROUP_NUM
from _path import generate_json_path,generate_pl_file,generate_cluster_file,generate_modified_cluster_file
import bin.parser.group_level_parser as parser
import numpy as np


test = parser.GroupLevelParser()
pl_path = generate_pl_file(BENCHMARK,DATASET)
cluster_path = generate_cluster_file(BENCHMARK,DATASET)
modified_cluster_path = generate_modified_cluster_file(BENCHMARK, DATASET)
net_list_path = generate_json_path(BENCHMARK, DATASET)[0]
pl_path = generate_pl_file(BENCHMARK, DATASET)
a = parser.load_net_list(net_list_path[:100])
b = parser.load_ori_cluster(cluster_path)
test.load_data(net_list_path, cluster_path)

a = test.modify_cluster(GROUP_NUM, pl_path, modified_cluster_path)
print(a)