import numpy as np

def read_data(path):
    """
    PAY ATTENTION: the path must obey the following rules:
    1. the end of the file must state "test" or "train" all lowercase!
    2. the end of the file must state "label" or "feature" all lowercase!
    For Example:
    "data/density/test1label.csv"   "data/density/trainlabel.csv"
    "data/density/test1feature.csv" "data/density/trainfeature.csv"
    start = path.rfind('/') # 字符串逆向查找
    start = path.find('/',__start=0,__end=len(path)) # 字符串正向查找
    """

    if path.rfind('test') > 0:  # read test data sets
        if path.rfind('label') > 0:
            path_label = path
            pathtmp = list(path)
            coor = path.rfind('label')
            pathtmp[coor:coor + len('label')] = 'feature'
            path_feature = ''.join(pathtmp)
            pathtmp2 = list(path)
            pathtmp2[coor:coor + len('label')] = 'adj'
            path_adj = ''.join(pathtmp2)

        elif path.rfind('feature') > 0:
            pathtmp = list(path)
            coor = path.rfind('feature')
            pathtmp[coor:coor + len('feature')] = 'label'
            path_label = ''.join(pathtmp)
            path_feature = path
            pathtmp2 = list(path)
            pathtmp2[coor:coor + len('feature')] = 'adj'
            path_adj = ''.join(pathtmp2)
    elif path.rfind('train') > 0:  # read train data sets
        if path.rfind('label') > 0:
            path_label = path
            pathtmp = list(path)
            coor = path.rfind('label')
            pathtmp[coor:coor + len('label')] = 'feature'
            path_feature = ''.join(pathtmp)
            pathtmp2 = list(path)
            pathtmp2[coor:coor + len('label')] = 'adj'
            path_adj = ''.join(pathtmp2)
        elif path.rfind('feature') > 0:
            pathtmp = list(path)
            coor = path.rfind('feature')
            pathtmp[coor:coor + len('feature')] = 'label'
            path_label = ''.join(pathtmp)
            path_feature = path
            pathtmp2 = list(path)
            pathtmp2[coor:coor + len('feature')] = 'adj'
            path_adj = ''.join(pathtmp2)
    else:
        raise ValueError("Wrong data set path! Can't find files.")

    label = np.loadtxt(path_label, delimiter=",")  # read as array
    feature = np.loadtxt(path_feature, delimiter=",")  # read as array
    # adj_matrix = np.loadtxt(path_adj, delimiter=',')
    adj_matrix = np.zeros(shape=(len(feature), len(feature[0])), dtype="double")
    connectivity_index = np.where(feature > 0)  # connectivity_index is a tuple data, each element is an array
    for i, j in zip(connectivity_index[0], connectivity_index[1]):
        adj_matrix[i][j] = 1
    for k in range(len(feature[0])):
        # print(len(feature))
        adj_matrix[k][k] = 1

    # adj_matrix = torch.from_numpy(adj_matrix)   # convert array to tensor
    return feature, adj_matrix, label
