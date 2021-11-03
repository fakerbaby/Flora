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


def generate_json_path(benchmark_name, dataset):
    _subnets_path = os.path.join(
        BASE_DIR, "tmp", benchmark_name, dataset, "bookshelf", dataset+"_subnet.json")
    _node2matrix_path = os.path.join(
        BASE_DIR, "tmp", benchmark_name, dataset, "bookshelf", dataset+"_.json")
    return _subnets_path, _node2matrix_path


def generate_cluster_file(benchmark_name, dataset):
    return os.path.join(BASE_DIR, "tmp", benchmark_name, dataset, "kmeans", dataset+".csv")


def generate_modified_cluster_file(benchmark_name, dataset):
    return os.path.join(BASE_DIR, "tmp", benchmark_name, dataset, "kmeans", dataset+"modified.csv")


def generate_pl_file(benchmark_name, dataset):
    return os.path.join(os.path.join(BASE_DIR, "benchmark",
                                     benchmark_name, dataset, dataset+".pl"))


net_path, node_path, first_mat_path = generate_path(BENCHMARK, DATASET)
pl_path = generate_pl_file(BENCHMARK, DATASET)
cluster_path = generate_cluster_file(BENCHMARK, DATASET)
modified_cluster_path = generate_modified_cluster_file(BENCHMARK, DATASET)
subnet_path, node2matrix_path = generate_json_path(BENCHMARK, DATASET)
