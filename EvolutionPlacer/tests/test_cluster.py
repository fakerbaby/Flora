# pylint: disable=import-error
from _const import GROUP_NUM
from _path import FIRST_MAT_PATH, ORI_CLUS_PATH
from bin.cluster.spectral_cluster import load_data, spectral_cluster


matrix_path = FIRST_MAT_PATH
cluster_file = ORI_CLUS_PATH

spectral_cluster(GROUP_NUM, matrix_path, cluster_file)