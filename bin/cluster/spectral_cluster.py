import time
from scipy.sparse.linalg import eigsh
from scipy.sparse import load_npz, csgraph
from sklearn.cluster import KMeans
from pathlib import Path


def load_data(sparse_matrix):
    """
    load sparse matrix from .npz file
    """
    return load_npz(sparse_matrix)


def spectral_cluster(group_number, sparse_matrix_file, kmeans_file):
    """
    spectral cluster to make the matrix produced by 'bookshelf_parser.py' into some clusters.
    """
    print("start spectral clustering...")
    sparse_matrix = load_data(sparse_matrix_file)
    laplacian_matrix = csgraph.laplacian(sparse_matrix, normed=True)
    eigen_value, eigen_vector = eigsh(laplacian_matrix, k=31, sigma=0, tol=0.01, which='LM')
    # eigenvalue, eigenvector = eigsh(laplacianMatrix, k=30, tol=0.1, which='SM')
    print("eigen_value:",eigen_value)
    kmeans = KMeans(n_clusters=group_number).fit(eigen_vector[:, 1:])
    kmeans_file.parent.mkdir(parents=True, exist_ok=True)  # create path if not exist
    kmeans.labels_.tofile(kmeans_file, sep=',')
