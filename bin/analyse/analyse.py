import csv
import pprint
import json
from pathlib import Path
import re
from matplotlib.pyplot import plot as plt
from scipy.sparse import load_npz


class Analyse:
    def __init__(self):
        self.data = []

    def cluster_analysis(self, filepath, savejsonfile):
        # read cluster .csv file as list
        clusterResultInfo = []
        with open(filepath, 'r')as f:
            csvReader = csv.reader(f)
            for nu in csvReader:
                for x in nu:
                    clusterResultInfo.append(int(x))

        # read cellName2Index json file
        # cellName2Index --> { cellseries : cell index }
        jsonFile = Path(Path.cwd(), 'result', 'cellName2Index.json')
        with open(jsonFile, 'r')as jsonf:
            jsoninfo = jsonf.read()
            cellName2Index = json.loads(jsoninfo)
            cellIndex2Name = {v: k for k, v in cellName2Index.items()}

        # save the real cell Name in def file into "NameDict" variables
        # NameDict-->{ cellseries : cell Name }
        componentNameFile = Path(Path.cwd(), '..', '40nm_testcase', '2_1_floorplan.def')
        with open(componentNameFile, 'r')as componentf:
            componentInfo = componentf.read()
            componentInfo = componentInfo[componentInfo.find('COMPONENTS'):componentInfo.find('END COMPONENTS')]
        NameDict = dict()
        for Name in cellName2Index:
            NameDict.setdefault(Name, '')
            subInfo = componentInfo[componentInfo.find(Name):]
            NameDict[Name] = subInfo[subInfo.find(Name):subInfo.find(';')].split(' ')[-2]

        # read cell size in .lef file and save it into "componentAreadict" variables
        # componentAreadict --> { cell Name : area }
        componentAreadict = dict()
        componentAreaFile = Path(Path.cwd(), '..', '40nm_testcase', 'scc40nll_vhsc40_rvt_ant.lef')
        with open(componentAreaFile, 'r')as componentAreaf:
            componentAreaInfo = componentAreaf.read()
        for componentName in NameDict.values():
            componentAreadict.setdefault(componentName, 0)
            macroInfo = componentAreaInfo[componentAreaInfo.find(componentName):componentAreaInfo.find('END '+componentName)]
            if re.search(r'SIZE\s(\d+.\d+)\sBY\s(\d+.\d+)\s', macroInfo) != None:
                length = re.search(r'SIZE\s(\d+.\d+)\sBY\s(\d+.\d+)\s', macroInfo).group(1)
                height = re.search(r'SIZE\s(\d+.\d+)\sBY\s(\d+.\d+)\s', macroInfo).group(2)
                componentAreadict[componentName] = float(length)*float(height)
            else:
                componentAreadict[componentName] = 255.69*213.595

        # "cellseries2area"    cellseries2area --> { cellseries : area }
        cellseries2area = {}
        for cells, celln in NameDict.items():
            cellArea = componentAreadict.get(celln, 0)
            cellseries2area.setdefault(cells, cellArea)

        clusterArea = dict()
        for clusterNo in range(max(clusterResultInfo)+1):
            No = [i for (i, v) in enumerate(clusterResultInfo) if v == clusterNo]
            area = 0
            for j in No:
                cellseries = cellIndex2Name.get(j, '')
                area = area + cellseries2area.get(cellseries, 0)
            clusterArea.setdefault(clusterNo, area)

        # count the number of instances for each cluster and record the total area for each cluster
        groupNumber = dict()
        for group in clusterResultInfo:
            groupNumber.setdefault(group, [0, 0])
            groupNumber[group][0] += 1
            groupNumber[group][1] = clusterArea[group]
        pprint.pprint(groupNumber)
        jsonstr = json.dumps(groupNumber)
        with open(savejsonfile, 'w')as ff:
            ff.write(jsonstr)

    def cell_area_check(self, file):
        area = 0
        with open(file, 'r')as f:
            info = f.read()
            info = json.loads(info)
            for i in info.values():
                area = area + i[1]
            print(f'{area}')

    def draw_each_group_info(self, file):
        groupIndex = list()
        eachCellNoinGroup = []
        eachCellAreainGroup = []
        with open(file, 'r')as f:
            info = json.load(f)
            for i,(j,k) in info.items():
                groupIndex.append(i)
                eachCellNoinGroup.append(j)
                eachCellAreainGroup.append(k)
        plt.scatter(groupIndex, eachCellNoinGroup)
        plt.show()

    def all_cell_area(self, designFile, libraryFile):
        macroArea = dict()
        with open(libraryFile, 'r')as f:
            libraryInfo = f.read()
            macroRegex = re.compile(r'MACRO')
            macroInfoList = re.split(macroRegex, libraryInfo)
            macroInfoList.pop(0)
            for macroLibraryInfo in macroInfoList:
                if re.search(r'\s*(\w+)\s', macroLibraryInfo) is not None:
                    macroName = re.search(r'\s*(\w+)\s', macroLibraryInfo).group(1)
                    res = re.search(r'SIZE\s*(\d+.\d+)\s*BY\s*(\d+.\d+)', macroLibraryInfo)
                    print(res.group())
                    if res is not None:
                        macro_area = round(float(res.group(1)) * float(res.group(2)), 4)
                        macroArea.setdefault(macroName, macro_area)
                    else:
                        print(f'{macroName} size is missing')
                else:
                    print(f'{macroLibraryInfo} didn\'t find macro')
            # add a special macro-area pair
            macroArea.setdefault('sram_sp_hde_x512y8d32_bw', round(float(54614.10555), 4))
        area = 0
        with open(designFile, 'r')as ff:
            designInfo = ff.read()
            componentList = designInfo[designInfo.find('COMPONENTS'):designInfo.find('END COMPONENTS')].split('\n')
            componentList.pop(-1)
            componentNumber2Name = dict()
            for i in componentList:
                componentNumber2Name.setdefault(i.rsplit(' ', 3)[-3], i.rsplit(' ', 2)[-2])
            componentNumber2Name.pop('COMPONENTS')
            for macro_name in componentNumber2Name.values():
                area = area + macroArea[macro_name]
            # for key in componentNumber2Name.keys():
            #     mes = len(re.findall(key.replace('\\', '\\\\\\'), designInfo))
            #     if mes >= 2:
            #         area = area + macroArea[componentNumber2Name[key]]
            #     else:
            #         print(f'{key} occur times wrong\n')
            print(f'total area is {area}')

    def check_sparseMatrix(self, sparseMatrixFile):
        sparseMatrix = load_npz(sparseMatrixFile)
        cellNameIndexFile = Path(sparseMatrixFile.parent, 'cellSeries2MatrixIndex.json')
        with open(cellNameIndexFile, 'r')as f:
            cellName2Index = json.load(f)
        # invert key and value in cellName2Index Dictionary
        Index2cellName = dict()
        for k, q in cellName2Index.items():
            Index2cellName.setdefault(q, k)
        for i in range(sparseMatrix.shape[0]):
            num_0 = 0
            for j in sparseMatrix[i, :].A[0]:
                if j == 0:
                    num_0 += 1
            if num_0 == sparseMatrix.shape[0]:
                print(f'{i}th row has 0 connected elements')
                print(f'{i}th row corresponds cell name is {Index2cellName[i]}')
            # set a breakpoint for easy analysis
            if i >= 1000:
                break

