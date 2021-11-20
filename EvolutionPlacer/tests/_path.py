import os
import sys
import random
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))  # the absolute path to base directory
sys.path.append(BASE_DIR)

benchmark = ["ispd2005"]
dataset = ["adaptec1", "adaptec2", "adaptec3", "adaptec4",
           "bigblue1", "bigblue2", "bigblue3", "bigblue4"]
model = ["twolayerGAT", "ResGAT", "ResGAT_noMLP"]

BENCHMARK = benchmark[0]  # choose the benchmark

DATASET = dataset[0]  # choose the dataset

MODEL = model[1]

# bookshelf related


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


# cluster related
def generate_ori_cluster_file(benchmark_name=BENCHMARK, dataset=DATASET):
    return os.path.join(BASE_DIR, "tmp", benchmark_name, dataset, "cluster", dataset+"_ori.csv")


def generate_extended_cluster_file(benchmark_name=BENCHMARK, dataset=DATASET):
    return os.path.join(BASE_DIR, "tmp", benchmark_name, dataset, "cluster", dataset+"_ext.csv")


def generate_modified_cluster_file(benchmark_name=BENCHMARK, dataset=DATASET):
    return os.path.join(BASE_DIR, "tmp", benchmark_name, dataset, "cluster", dataset+"_mod.csv")


def generate_result_cluster_path(benchmark_name=BENCHMARK, dataset=DATASET):
    return os.path.join(BASE_DIR, "tmp", benchmark_name, dataset, "cluster", dataset+"_result1.csv")


# others
def generate_adjacent_path(benchmark_name=BENCHMARK, dataset=DATASET):
    return os.path.join(BASE_DIR, "tmp", benchmark_name, dataset, "grouplevel", dataset+"_adj.csv")


def generate_feature_mat_path(benchmark_name=BENCHMARK, dataset=DATASET):
    return os.path.join(BASE_DIR, "tmp", benchmark_name, dataset, "grouplevel", dataset+"_fea_mat.csv")


def generate_pl_file(benchmark_name=BENCHMARK, dataset=DATASET):
    return os.path.join(os.path.join(BASE_DIR, "benchmark",
                                     benchmark_name, dataset, dataset+".pl"))


# model realted
def get_model_data_path(Type: str, benchmark_name=BENCHMARK, dataset=DATASET):
    distance = os.path.join(os.path.join(BASE_DIR, "data", benchmark_name, dataset, str(Type) +
                                         "_distance.csv"))
    feature = os.path.join(os.path.join(BASE_DIR, "data", benchmark_name, dataset, str(Type) +
                                        "_feature.csv"))
    label = os.path.join(os.path.join(BASE_DIR, "data", benchmark_name, dataset, str(Type) +
                                      "_label.csv"))
    return distance, feature, label


def get_model_attn_path(benchmark_name=BENCHMARK, dataset=DATASET):
    attn = os.path.join(os.path.join(BASE_DIR, "tmp",  benchmark_name, dataset, "model_saved",
                                     "GAT2_model_"+str(random.randint(0, 10000))))
    pl = attn + ".pt"
    csv = attn + ".csv"
    return pl, csv


# bookshelf related path
NET_PATH, NODE_PATH, FIRST_MAT_PATH = generate_path()
SUB_NET_PATH, NODE2MAT_PATH = generate_json_path()

# cluster related path
ORI_CLUS_PATH = generate_ori_cluster_file()
EXT_CLUS_PATH = generate_extended_cluster_file()
MOD_CLUS_PATH = generate_modified_cluster_file()
RES_CLUS_PATH = generate_result_cluster_path()
# others
PL_PATH = generate_pl_file()
ADJ_PATH = generate_adjacent_path()
FEAT_MAT_PATH = generate_feature_mat_path()


# model
DATA_FILE_PATH = os.path.join(BASE_DIR, "data", BENCHMARK, DATASET)
