import os
import time
from scipy.sparse.linalg import eigsh
from scipy.sparse import load_npz, csgraph
from sklearn.cluster import KMeans

def load_data(sparse_matrix):
    """
    load sparse matrix from .npz file
    """
    return load_npz(sparse_matrix)


def spectral_cluster(group_number, sparse_matrix_file, kmeans_file):
    """
    spectral cluster to make the matrix produced by 'bookshelf_parser.py' into some clusters.
    Args:
        group_number: 
        sparse_matrix_file:
        kmeans_file:
    Returns:
    """
    print("="*18,"clustering start","="*18)
    print("loading matrix...")
    start_time = time.time()
    sparse_matrix = load_data(sparse_matrix_file)
    print("establishing a laplacianmatrix...")
    laplacian_matrix = csgraph.laplacian(sparse_matrix, normed=True)
    print("calculating eigen value and eigen vector...")
    eigen_value, eigen_vector = eigsh(laplacian_matrix, k=31, tol=0.1, which='LM')
    print("use Kmeans algorithm to fit...")
    kmeans = KMeans(n_clusters=group_number).fit(eigen_vector[:, 1:]) #the least eigen value is 0
    end_time = time.time() - start_time
    print("spectral clustering time consuming is {time}s".format(time = end_time))
    if not os.path.exists(os.path.dirname(kmeans_file)):
        os.makedirs(os.path.dirname(kmeans_file))
    else:
        pass
    print("saving data...")
    kmeans.labels_.tofile(kmeans_file, sep=',')
    print("saved")
    print("="*18,"clustering end","="*18)
    
