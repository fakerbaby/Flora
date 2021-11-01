from pathlib import Path
from analyse.analyse import Analyse
from parser.defparser import Parser
from cluster.spectrumcluster import Cluster
import time

if __name__ == '__main__':
    # *******************************************************
    #                           FIRST
    # This part should be replaced by CLI or GUI
    # *******************************************************
    PPPDef = Parser()
    datasetName = 'ispd2014'
    chipName = 'mgc_des_perf_2'
    basePath = Path(Path.cwd(), 'result', datasetName, chipName)
    netFilepath = Path(Path.cwd(), '..', 'benchmark', datasetName, chipName, 'floorplan.def')
    group_number = 625

    # ******************************************************
    #                           SECOND
    # Read netlist file and extract the cells' connectivity info
    # ******************************************************
    print("EvolutionPlacer has been boot...")
    saveCellName2MatrixIndexFile = Path(basePath, 'cellName2MatrixIndex.json')
    saveSparseMatrixFile = Path(basePath, 'original_sparse_matrix.npz')
    start = time.time()
    # PPPDef.net2sparsematrix(netFilepath, saveSparseMatrixFile, saveCellName2MatrixIndexFile)
    end = time.time()
    print(f"All cells' connectivity sparse matrix has been constructed. Spend {end-start}s.")

    # ********************************************************
    #                           THIRD
    # Cluster the cells into different groups
    # ********************************************************
    print("Waiting for clustering...")
    PPPCluster = Cluster()
    saveClusterFile = Path(basePath, 'cluster_result'+str(group_number)+'.csv')
    PPPCluster.spectral_cluster(group_number, saveSparseMatrixFile, saveClusterFile)
    start = time.time()
    # PPPCluster.spectral_cluster(group_number, saveSparseMatrixFile, saveClusterFile)
    end = time.time()
    print(f"The clustering process has completed. Spend {end-start}s.")

    # ***************************************************************
    #                           FOURTH
    # Get group-level connectivity information based on cluster results
    # ***************************************************************
    print("Waiting for group-level connectivity features...")
    saveGroupLevelConnectivityFile = Path(basePath, str(group_number)+'groupConnectivity'+'.csv')
    start = time.time()
    # PPPDef.group_level_connection(saveClusterFile, netFilepath, saveCellName2MatrixIndexFile, saveGroupLevelConnectivityFile)
    # PPPDef.group_level_feature(saveSparseMatrixFile, saveClusterFile, saveGroupLevelConnectivityFile)
    end = time.time()
    print(f"The group-level connectivity features has completed. Spend {end-start}s.")

    # ***************************************************************
    #                           FIFTH
    # load GAT model and get the position results
    # TODO : our model only supports 900 groups right now
    #  please  refer  loadModel-icbench.py
    #  Models that support various groups coming soon
    # ***************************************************************

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
    print("Waiting for writing position info...")
    print("\t\tNOTE:\tThe position file has been saved to benchmark directory instead of result!")
    positionFile = Path(basePath, str(group_number)+'groups-position.pl')
    initialDefFile = netFilepath
    resDefFile = Path(initialDefFile.parent, str(group_number)+'groups-position.def')
    start = time.time()
    PPPDef.write_group_position(initialDefFile, positionFile, resDefFile)
    end = time.time()
    print(f"""The group-level position file has completed. Spend {end-start}s\n\
    Try to use Dreamplace to get finial placement.\n""")

    #  **************************************************************
    #  对结果正确性的分析代码，真正运行时无需考虑
    #  **************************************************************
    a = Analyse()
    # clusterFile = Path(Path.cwd(), 'result', datasetName, chipName, 'cluster_result'+str(group_number)+'.csv')
    # eachGroupInfoFile = Path(Path.cwd(), 'result', datasetName, chipName, 'group_level_info.json')
    # a.check_sparseMatrix(saveSparseMatrixFile)
    # a.cluster_analysis(clusterFile, eachGroupInfoFile)
    # a.draw_each_group_info(eachGroupInfoFile)

    # designFilePath = Path(Path.home(), 'Projects', '40nm_testcase', '2_1_floorplan.def')
    # libraryFilePath = Path(Path.home(), 'Projects', '40nm_testcase', 'scc40nll_vhsc40_rvt_ant.lef')
    # a.all_cell_area(designFilePath, libraryFilePath)
    # a.cell_area_check(Path('result', 'group_level_info.json'))
