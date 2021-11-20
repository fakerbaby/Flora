import _path as path
import _const as const
import torch
from torch import nn, optim
import numpy as np
import bin.model.model as mm
import bin.model.model_train as mmt
from bin.model.load_data import read_data



model = mmt.ModelTrain()

#model1
# model.load_data(path.DATA_FILE_PATH)
# model.train_and_test_distance_model()
# model.train_and_test_position_model()
# model.load_final_data(path.DATA_FILE_PATH)
# model.load_two_layer_model()

#ResGAT
# model.load_data(path.DATA_FILE_PATH)
# model.train_res_gat_model()

# model.load_final_data(path.DATA_FILE_PATH)
# model.load_res_model()
# model.load_final_data(path.DATA_FILE_PATH)
# model.load_res_model_without_distance()

#ResGAT_without_MLP
# model.load_data(path.DATA_FILE_PATH)
# model.train_res_gat_model_noMLP()

model.load_final_data(path.DATA_FILE_PATH)
model.load_res_model_noMLP()



# model.load_final_data_without_distance(path.DATA_FILE_PATH)
# model.load_res_model_without_distance_noMLP()

