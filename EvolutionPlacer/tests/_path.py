import os
from _const import BASE_DIR, BENCHMARK, DATASET


def generate_path(benchmark_name, dataset):
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


def generate_pl_file(benchmark_name, dataset):
    return os.path.join(os.path.join(BASE_DIR, "benchmark",
                                     benchmark_name, dataset, dataset+".pl"))


def generate_json_path(benchmark_name, dataset):
    _subnets_path = os.path.join(
        BASE_DIR, "tmp", benchmark_name, dataset, "bookshelf", dataset+"_subnet.json")
    _node2matrix_path = os.path.join(
        BASE_DIR, "tmp", benchmark_name, dataset, "bookshelf", dataset+"_.json")
    return _subnets_path, _node2matrix_path


def generate_ori_cluster_file(benchmark_name, dataset):
    return os.path.join(BASE_DIR, "tmp", benchmark_name, dataset, "cluster", dataset+"_ori.csv")


def generate_modified_cluster_file(benchmark_name, dataset):
    return os.path.join(BASE_DIR, "tmp", benchmark_name, dataset, "cluster", dataset+"_mod.csv")


def generate_adjacent_path(benchmark_name, dataset):
    return os.path.join(BASE_DIR, "tmp", benchmark_name, dataset, "grouplevel", dataset+"_adj.csv" )


def generate_feature_mat_path(benchmark_name, dataset):
    return os.path.join(BASE_DIR, "tmp", benchmark_name, dataset, "grouplevel", dataset+"_fea_mat.csv" )


NET_PATH, NODE_PATH, FIRST_MAT_PATH = generate_path(BENCHMARK, DATASET)
PL_PATH = generate_pl_file(BENCHMARK, DATASET)
ORI_CLUS_PATH = generate_ori_cluster_file(BENCHMARK, DATASET)
MOD_CLUSTER_PATH = generate_modified_cluster_file(BENCHMARK, DATASET)
SUB_NET_PATH, NODE2MAT_PATH = generate_json_path(BENCHMARK, DATASET)
ADJ_PATH = generate_adjacent_path(BENCHMARK, DATASET)
FEAT_MAT_PAT = generate_feature_mat_path(BENCHMARK, DATASET)
