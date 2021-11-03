# pylint: disable=import-error
from _const import BENCHMARK, DATASET, GROUP_NUM
from _path import generate_path, generate_cluster_file
from bin.cluster.spectral_cluster import load_data, spectral_cluster


matrix_path = generate_path(BENCHMARK,DATASET)[2]
cluster_file = generate_cluster_file(BENCHMARK, DATASET)
test = load_data(matrix_path)

spectral_cluster(GROUP_NUM, matrix_path, cluster_file)