from pathlib import Path
from random import choices
from tests._const import *
from tests._path import *
# from analyse.analyse import Analyse
from bin.parser.bookshelf_parser import BookshelfParser
from bin.parser.group_level_parser import GroupLevelParser
from bin.cluster.spectral_cluster import spectral_cluster
import bin.model.model as mm
import bin.model.model_train as mmt
from bin.model.load_data import read_data

import time

if __name__ == '__main__':
    # *******************************************************
    #                           FIRST
    # This part should be replaced by CLI or GUI
    # *******************************************************
    print("choose the  data type")
    print("1.real-world data 2.random-generated data")
    choice = int(input())

    BSParser = BookshelfParser()
    net_path, node_path, target_path = NET_PATH, NODE_PATH, FIRST_MAT_PATH
    subnet_path, node2matrix_path = SUB_NET_PATH, NODE2MAT_PATH

    # ******************************************************
    #                           SECOND
    # Read netlist file and extract the cells' connectivity info
    # ******************************************************
    print("EvolutionPlacer has been boot...")
    start = time.time()
    BSParser.load_data(net_path, node_path)
    BSParser.net_to_matrix(target_path, node2matrix_path, subnet_path)
    BSParser.monitor
    # saveCellName2MatrixIndexFile = Path(basePath, 'cellName2MatrixIndex.json')
    # saveSparseMatrixFile = Path(basePath, 'original_sparse_matrix.npz')
    # PPPDef.net2sparsematrix(netFilepath, saveSparseMatrixFile, saveCellName2MatrixIndexFile)
    end = time.time()
    print(f"All cells' connectivity sparse matrix has been constructed. Spend {end - start}s.")

    # ********************************************************
    #                           THIRD
    # Cluster the cells into different groups
    # ********************************************************
    print("Waiting for clustering...")

    matrix_path = FIRST_MAT_PATH
    cluster_file = ORI_CLUS_PATH
    start = time.time()
    PPPCluster = spectral_cluster(GROUP_NUM, FIRST_MAT_PATH, ORI_CLUS_PATH)
    end = time.time()
    print(f"The clustering process has completed. Spend {end - start}s.")

    # ***************************************************************
    #                           FOURTH
    # Get group-level connectivity information based on cluster results
    # ***************************************************************
    print("Waiting for group-level connectivity features...")
    GLParser = GroupLevelParser()
    start = time.time()
    GLParser.load_data(SUB_NET_PATH, ORI_CLUS_PATH)
    GLParser.extend_cluster(PL_PATH, GROUP_NUM)
    GLParser.cluster_fusion()
    GLParser.add_auxiliary_conncetivity(THRESHOLD, ONE_WEIGHT, TWO_WEIGHT)
    GLParser.save_data(EXT_CLUS_PATH, ADJ_PATH, FEAT_MAT_PATH, RES_CLUS_PATH)
    # PPPDef.group_level_connection(saveClusterFile, netFilepath, saveCellName2MatrixIndexFile, saveGroupLevelConnectivityFile)
    # PPPDef.group_level_feature(saveSparseMatrixFile, saveClusterFile, saveGroupLevelConnectivityFile)
    end = time.time()
    print(f"The group-level connectivity features has completed. Spend {end - start}s.")

    # ***************************************************************
    #                           FIFTH
    # load GAT model and get the position results
    # TODO : our model only supports 900 groups right now
    #  please  refer  loadModel-icbench.py
    #  Models that support various groups coming soon
    # ***************************************************************
    model = mmt.ModelTrain()
    model.load_data(DATA_FILE_PATH)
    model.train_and_test_distance_model()
    model.train_and_test_position_model()
    model.load_final_data(DATA_FILE_PATH)
    model.load_two_layer_model()

    # ***************************************************************
    #                           SIXTH
    # generate all macro position based on GAT
    # TODO:
    #   please refer AssignPos.py
    # ***************************************************************

    # ***************************************************************
    #                           SEVENTH
    # transform the position result to del file
    # ***************************************************************
    # print("Waiting for writing position info...")
    # print("\t\tNOTE:\tThe position file has been saved to benchmark directory instead of result!")
    # positionFile = Path(basePath, str(group_number)+'groups-position.pl')
    # initialDefFile = netFilepath
    # resDefFile = Path(initialDefFile.parent, str(group_number)+'groups-position.def')
    # start = time.time()
    # PPPDef.write_group_position(initialDefFile, positionFile, resDefFile)
    # end = time.time()
    # print(f"""The group-level position file has completed. Spend {end-start}s\n\
    # Try to use Dreamplace to get finial placement.\n""")

    #  **************************************************************
    #  对结果正确性的分析代码，真正运行时无需考虑
    #  **************************************************************
    # a = Analyse()
    # clusterFile = Path(Path.cwd(), 'result', datasetName, chipName, 'cluster_result'+str(group_number)+'.csv')
    # eachGroupInfoFile = Path(Path.cwd(), 'result', datasetName, chipName, 'group_level_info.json')
    # a.check_sparseMatrix(saveSparseMatrixFile)
    # a.cluster_analysis(clusterFile, eachGroupInfoFile)
    # a.draw_each_group_info(eachGroupInfoFile)

    # designFilePath = Path(Path.home(), 'Projects', '40nm_testcase', '2_1_floorplan.def')
    # libraryFilePath = Path(Path.home(), 'Projects', '40nm_testcase', 'scc40nll_vhsc40_rvt_ant.lef')
    # a.all_cell_area(designFilePath, libraryFilePath)
    # a.cell_area_check(Path('result', 'group_level_info.json'))
