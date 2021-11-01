import re
import json
import time
from pathlib import Path
from scipy.sparse import csc_matrix, lil_matrix, save_npz
from scipy.sparse.linalg import eigsh
from scipy.sparse import load_npz, csgraph
from sklearn.cluster import KMeans
import numpy as np
import csv


class Parser:
    def __init__(self):
        data = []

    def net2sparsematrix(self, netfilepath, sparseMatrixFile, cellName2MatrixIndex):
        """
        read floorplan.def file, extract all cells' connectivity relationship, save it into sparse matrix.
        There is also need to construct a dictionary which records the cell name to matrix Index.
        (Sort according to the order in which the cell appear)

        :param netfilepath: Path of netlist file.
        :param sparseMatrixFile: Path where the original sparse matrix will be stored.
        :param cellName2MatrixIndex: Path where the cellName2MatrixIndex dictionary will be stored.
        :return:
        """
        with open(netfilepath, 'r') as f:
            info = f.read()
            # read NET file
            netRegex = re.compile(r'\bnets\s+\d+\s*;', re.IGNORECASE) # \\b matches the empty string, but only at the beginning or end of a word
            endNetRegex = re.compile(r'\bend\s+nets\s+', re.IGNORECASE)
            # netRegex = re.compile(r'^nets\s+\d+\s*', re.IGNORECASE, re.MULTILINE)
            # endNetRegex = re.compile(r'^end\s+nets\s+', re.IGNORECASE,re.MULTILINE)
            netInfo = info[info.find(re.search(netRegex, info).group()):info.find(re.search(endNetRegex, info).group())]
            # read cell info
            totalCellNumber = int(re.search(r'COMPONENTS\s(\d+)\s;', info).group(1))
            cellInfo = info[info.find('COMPONENTS'):info.find('END COMPONENTS')]
            cellList = re.split(r';', cellInfo)
            cellList.pop(0)  # delete 'COMPONENTS xxxx'
            cellList.pop(-1)  # delete '\n'
            # construct cellName2MatrixIndex dictionary
            cellName2Index = dict()
            for i in range(totalCellNumber):
                CellName = re.split(r'\s', re.search(r'-\s+(.*)', cellList[i]).group(1))[0]
                cellName2Index.setdefault(CellName, i)
            # read PIN(IO Pad) info
            PINSRegex = re.compile(r'pins\s+(\d+)', re.IGNORECASE)
            totalPinNumber = int(re.search(PINSRegex, info).group(1))
            PINInfo = info[info.find('PINS'):info.find('END PINS')]
            PINList = re.split(r';', PINInfo)
            PINList.pop(0)
            PINList.pop(-1)
            for i in range(totalPinNumber):
                PINName = re.search(r'-\s+(\w+)\s+', PINList[i]).group(1)
                cellName2Index.setdefault(PINName, i+totalCellNumber)
        # save cellSeries2MatrixIndex dict into Json file
        cellName2MatrixIndex.parent.mkdir(parents=True, exist_ok=True)  # create path if not exist
        json_str = json.dumps(cellName2Index)
        with open(cellName2MatrixIndex, 'w') as ff:
            ff.write(json_str)

        # ****************
        #  make regex
        # ****************
        subNetRegex = re.compile(r'-\s(.*?)\s')
        connectCellRegex = re.compile(r'\(\s+(.*?)\s+(.*?)\s+\)')
        # numberConnectCellRegex = re.compile(r'\(\s_(\d+)_\s\w+\s\)')

        #  loop the NET file content and find the regex
        subNetNo, cellName = [], []
        netInfoList = re.split(r';', netInfo)
        for subNetInfo in netInfoList:
            NetName = re.search(subNetRegex, subNetInfo)
            if NetName is not None:
                subNetNo.append(NetName.group(1))
            connectCell = re.findall(connectCellRegex, subNetInfo)
            if len(connectCell) != 0:
                cellName.append(connectCell)
        
        #  make sparse matrix
        sparseMatrix = lil_matrix((len(cellName2Index), len(cellName2Index)))
        for component in cellName:
            adjacentcell = []
            for x, y in component:
                if x == 'PIN':
                    adjacentcell.append(y)
                else:
                    adjacentcell.append(x)
            rootcell = adjacentcell[0]
            connectedcell = adjacentcell[1:]
            for xxx in connectedcell:
                sparseMatrix[cellName2Index[rootcell], cellName2Index[xxx]] += 1
                sparseMatrix[cellName2Index[xxx], cellName2Index[rootcell]] += 1
        digIndex = [i for i in range(len(cellName2Index))]
        sparseMatrix[digIndex, digIndex] = 0
        sparseMatrix = csc_matrix(sparseMatrix)

        # save results
        save_npz(sparseMatrixFile, sparseMatrix)

    def group_level_connection(self, clusterResultFile, NetFile, cellName2IndexFile, groupConnectivityFile):
        """

        :param clusterResultFile:
        :param NetFile:
        :param cellName2IndexFile:
        :param groupConnectivityFile:
        :return:
        """
        with open(clusterResultFile, 'r')as f:
            clusterResult = csv.reader(f)
            clusterResultList = []
            for i in clusterResult:
                for x in i:
                    clusterResultList.append(int(x))
        with open(NetFile, 'r')as ff:
            NetInfo = ff.read()
            netRegex = re.compile(r'\bnets\s+\d+\s*;',re.IGNORECASE)  # \\b matches the empty string, but only at the beginning or end of a word
            endNetRegex = re.compile(r'\bend\s+nets\s+', re.IGNORECASE)
            NetInfo = NetInfo[NetInfo.find(re.search(netRegex, NetInfo).group()):NetInfo.find(re.search(endNetRegex, NetInfo).group())]
        with open(cellName2IndexFile, 'r')as fff:
            strings = fff.read()
            cellName2Index = json.loads(strings)

        # create a matrix size of groups
        size = max(clusterResultList) + 1
        groupMatrix = np.zeros(shape=(size, size), dtype='int')

        # loop subnet in NET info
        connectCellRegex = re.compile(r'\(\s+(.*?)\s+(.*?)\s+\)')
        NetList = re.split(r';', NetInfo)
        NetList.pop(0)
        NetList.pop(-1)
        for subnet in NetList:
            connectedCell = re.findall(connectCellRegex, subnet)
            adjacentCell = []
            for x, y in connectedCell:
                if x == 'PIN':
                    adjacentCell.append(y)
                else:
                    adjacentCell.append(x)
            rootCell = adjacentCell[0]
            connectedcell = adjacentCell[1:]
            rootCellGroup = clusterResultList[cellName2Index[rootCell]]
            for cell in connectedcell:
                adjacentCellGroup = clusterResultList[cellName2Index[cell]]
                if rootCellGroup != adjacentCellGroup:
                    groupMatrix[rootCellGroup, adjacentCellGroup] += 1
                    groupMatrix[adjacentCellGroup, rootCellGroup] += 1
                else:
                    pass

            # if connectedCell[0] == 'PIN':
            #     if len(connectedCell) > 2:
            #         rootCell = connectedCell[1]
            #         adjacentCell = connectedCell[2:]
            #         for cell in adjacentCell:
            #             rootCellGroup = clusterResultList[cellName2Index[rootCell]]
            #             adjacentCellGroup = clusterResultList[cellName2Index[cell]]
            #             if rootCellGroup != adjacentCellGroup:
            #                 groupMatrix[rootCellGroup, adjacentCellGroup] += 1
            #                 groupMatrix[adjacentCellGroup, rootCellGroup] += 1
            #             else:
            #                 pass
            # else:
            #     rootCell = connectedCell[0]
            #     rootCellGroup = clusterResultList[cellName2Index[rootCell]]
            #     adjacentCell = connectedCell[1:]
            #     for cell in adjacentCell:
            #         adjacentCellGroup = clusterResultList[cellName2Index[cell]]
            #         if rootCellGroup != adjacentCellGroup:
            #             groupMatrix[rootCellGroup, adjacentCellGroup] += 1
            #             groupMatrix[adjacentCellGroup, rootCellGroup] += 1
            #         else:
            #             pass
        # save group connectivity matrix
        np.savetxt(groupConnectivityFile, groupMatrix, fmt='%d', delimiter=',')

    def group_level_feature(self, originalSparseMatrixFile, clusterResultFile, groupConnectivityFile):
        """

        :param originalSparseMatrixFile:
        :param clusterResultFile:
        :param groupConnectivityFile:
        :return:
        """
        sparseMatrix = load_npz(originalSparseMatrixFile)
        # read csv file
        with open(clusterResultFile, 'r')as f:
            clusterResult = csv.reader(f)
            clusterResultList = []
            for header in clusterResult:
                for x in header:
                    clusterResultList.append(int(x))
        clusterResult = np.array(clusterResultList)
        del clusterResultList
        size = max(clusterResult) + 1
        groupMatrix = np.zeros(shape=(size, size), dtype='int')
        for i in range(sparseMatrix.shape[0]):
            connectCell = sparseMatrix[i, :].nonzero()[1]
            rootCell = i
            for cell in connectCell:
                rootGroup = clusterResult[rootCell]
                cellGroup = clusterResult[cell]
                if rootGroup != cellGroup:
                    groupMatrix[rootGroup, cellGroup] += 1
                    groupMatrix[cellGroup, rootGroup] += 1
                else:
                    pass
        # save group connectivity matrix
        np.savetxt(groupConnectivityFile, groupMatrix, fmt='%d', delimiter=',')

        # ***************************
        # Loop through the whole sparse matrix is too time-consuming
        # It needs to scan a matrix size totalCellNum * totalCellNum 
        # ***************************
        # groupDict = dict()
        # for i in range(size):
        #     element = np.where(clusterResult == i)[0]
        #     groupDict.setdefault(i, element)
        # groupMatrix = np.zeros(shape=(size, size), dtype='int')
        # groupList = [i for i in groupDict.keys()]
        # for i in groupList[:-1]:
        #     for j in groupList[i+1:]:
        #         num = 0
        #         for x in groupDict[i]:
        #             for y in groupDict[j]:
        #                 num += sparseMatrix[x, y]
        #         groupMatrix[i, j] = num
        #         groupMatrix[j, i] = num
        # # save group connectivity matrix
        # np.savetxt(groupConnectivityFile, groupMatrix, fmt='%d', delimiter=',')

    def writeGroupPosition(self, defFile, positionFile, saveFile):
        """

        :param defFile:
        :param positionFile:
        :param saveFile:
        :return:
        """
        positionList = []
        with open(positionFile, 'r')as postionf:
            while True:
                info = postionf.readline()
                if not info:
                    print(f'\t\tprogram finished!')
                    break
                if re.search(r'[:;]\s*[a-zA-Z]', info) is not None:
                    if re.search(r'\d+.\d+\s+\d+.\d+\s+:', info) is not None:
                        position = re.search(r'(\d+.\d+\s+\d+.\d+\s+):', info).group(1)
                        complement = re.search(r'[:;](\s*[a-zA-Z])', info).group(1)
                        positionList.append([position, complement])
        with open(defFile, 'r')as f:
            info = f.read()
            componentsNum = re.search(r'COMPONENTS\s+\d+\s+;', info).group()
            componentsInfoList = info[info.index('COMPONENTS'):info.index('END COMPONENTS')].split('\n')
            componentsInfoList.pop(0)
            componentsInfoList.pop(-1)
        result = ''
        result += info[0:info.index('COMPONENTS')]
        result += componentsNum+'\n'
        for componentOriginalInfo, pos in zip(componentsInfoList, positionList):
            result += componentOriginalInfo.replace(';', '') + ' + PLACED ( ' + pos[0] + ')' + pos[1]+' ;\n'
        result += info[info.index('END COMPONENTS'):]
        with open(saveFile, 'w')as ff:
            ff.write(result)

    def write_group_positions(self, defFile, positionFile, saveFile):
        """

        :param defFile:
        :param positionFile:
        :param saveFile:
        :return:
        """
        with open(defFile, 'r')as f:
            deffile = f.read()

            # read movable cell number
            componentsRegex = re.compile(r'\bCOMPONENTS\s+(\d+)\s*;')
            Num = re.findall(componentsRegex, deffile)
            movableCellNum = 0
            for i in Num:
                movableCellNum += int(i)

            # delete REGIONS
            regionsRegex = re.compile(r'\bREGIONS\s+\d+\s*;', re.IGNORECASE)
            endregionsRegex = re.compile(r'\bEND\s+REGIONS\s*', re.IGNORECASE)
            regionp = re.findall(regionsRegex, deffile)
            endregionp = re.findall(endregionsRegex, deffile)
            for i, j in zip(regionp, endregionp):
                deffile = deffile.replace(deffile[deffile.find(i):deffile.find(j)], '')
                # You're probably best off using the regular expression sub method with the re.IGNORECASE option.
                deffile = deffile.replace('END REGIONS', '')

            # delete GROUPS
            gRegex = re.compile(r'\bGROUPS\s+\d+\s*;')
            endgRegex = re.compile(r'\bEND\s+GROUPS\s+\d+\s*;')
            gp = re.findall(gRegex, deffile)
            endgp = re.findall(endgRegex, deffile)
            for i, j in zip(gp, endgp):
                deffile = deffile.replace(deffile[deffile.find(i):deffile.find(j)], '')
                deffile = deffile.replace('END GROUPS', '')

        with open(positionFile, 'r')as ff:
            positionInfo = ff.read()
            positionRegex = re.compile(r'o\d+\s+(\d+)\s+(\d+)\s+:\s+N')
            position = re.findall(positionRegex, positionInfo)
            movableCellPos = position[:movableCellNum]
        for pos_x, pos_y in movableCellPos:
            # unplaceIndex = deffile.find('UNPLACED')
            deffile = deffile.replace('UNPLACED', 'PLACED ( '+ pos_x + ' ' + pos_y +' ) N', 1)

        with open(saveFile, 'w')as fff:
            fff.write(deffile)

    def write_group_position(self, defFile, positionFile, saveFile):
        """

        :param defFile: floorplan.def file in which contains cells' position information
        :param positionFile: new position info of each cell generated by GAT model
        :param saveFile: path of new floorplan.def in which contains new positions generated from GAT
        :return:
        """