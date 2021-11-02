import os
from _const import BASE_DIR


def generate_path(benchmark_name, dataset):
    """
    generate the path to benchmarks
    """
    net_path = os.path.join(BASE_DIR,"benchmark",
                            benchmark_name, dataset, dataset+".nets")
    node_path = os.path.join(BASE_DIR, "benchmark",
                            benchmark_name, dataset, dataset+".nodes")
    target_path = os.path.join(BASE_DIR, "tmp",
                                benchmark_name, dataset,"bookshelf", dataset+".npz")
    return net_path, node_path, target_path


def generate_json_path(benchmark_name, dataset):
    return os.path.join(BASE_DIR,"tmp", benchmark_name, dataset,"bookshelf", dataset+".json")

        

    