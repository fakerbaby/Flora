import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  #the absolute path to base directory
sys.path.append(BASE_DIR)

benchmark = ["ispd2005"]
dataset = ["adaptec1", "adaptec2","adaptec3", "adaptec4", "bigblue1", "bigblue2", "bigblue3", "bigblue4"]


BENCHMARK = benchmark[0] # choose the benchmark

DATASET = dataset[0] # choose the dataset


#bookshelf related
def generate_path(benchmark_name=BENCHMARK, dataset=DATASET):
    """
    generate the path to benchmarks
    """
    _net_path = os.path.join(BASE_DIR, "benchmark",
                             benchmark_name, dataset, dataset+".nets")
    _node_path = os.path.join(BASE_DIR, "benchmark",
                              benchmark_name, dataset, dataset+".nodes")
    _first_mat_path = os.path.join(BASE_DIR, "tmp",
                                   benchmark_name, dataset, "bookshelf", dataset+".npz")
    return _net_path, _node_path, _first_mat_path

def generate_json_path(benchmark_name=BENCHMARK, dataset=DATASET):
    _subnets_path = os.path.join(
        BASE_DIR, "tmp", benchmark_name, dataset, "bookshelf", dataset+"_subnet.json")
    _node2matrix_path = os.path.join(
        BASE_DIR, "tmp", benchmark_name, dataset, "bookshelf", dataset+"_.json")
    return _subnets_path, _node2matrix_path


#cluster related 
def generate_ori_cluster_file(benchmark_name=BENCHMARK, dataset=DATASET):
    return os.path.join(BASE_DIR, "tmp", benchmark_name, dataset, "cluster", dataset+"_ori.csv")

def generate_extended_cluster_file(benchmark_name=BENCHMARK, dataset=DATASET):
    return os.path.join(BASE_DIR, "tmp", benchmark_name, dataset, "cluster", dataset+"_ext.csv")

def generate_modified_cluster_file(benchmark_name=BENCHMARK, dataset=DATASET):
    return os.path.join(BASE_DIR, "tmp", benchmark_name, dataset, "cluster", dataset+"_mod.csv")

def generate_result_cluster_path(benchmark_name=BENCHMARK, dataset=DATASET):
    return os.path.join(BASE_DIR, "tmp", benchmark_name, dataset, "cluster", dataset+"_result1.csv" )

#others
def generate_adjacent_path(benchmark_name=BENCHMARK, dataset=DATASET):
    return os.path.join(BASE_DIR, "tmp", benchmark_name, dataset, "grouplevel", dataset+"_adj.csv" )

def generate_feature_mat_path(benchmark_name=BENCHMARK, dataset=DATASET):
    return os.path.join(BASE_DIR, "tmp", benchmark_name, dataset, "grouplevel", dataset+"_fea_mat.csv" )

def generate_pl_file(benchmark_name=BENCHMARK, dataset=DATASET):
    return os.path.join(os.path.join(BASE_DIR, "benchmark",
                                     benchmark_name, dataset, dataset+".pl"))



#bookshelf related path
NET_PATH, NODE_PATH, FIRST_MAT_PATH = generate_path()
SUB_NET_PATH, NODE2MAT_PATH = generate_json_path()

#cluster related path
ORI_CLUS_PATH = generate_ori_cluster_file()
EXT_CLUS_PATH = generate_extended_cluster_file()
MOD_CLUS_PATH = generate_modified_cluster_file()
RES_CLUS_PATH = generate_result_cluster_path()
#others
PL_PATH = generate_pl_file()
ADJ_PATH = generate_adjacent_path()
FEAT_MAT_PATH = generate_feature_mat_path()
