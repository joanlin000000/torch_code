#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import yaml
import argparse
import torch
from model import YNet





CONFIG_FILE_PATH = 'C:/Users/abcd/Human-Path-Prediction-master/YNet/Human-Path-Prediction-master/ynet/ynet_additional_files/config/sdd_longterm.yaml'  # yaml config file containing all the hyperparameters
EXPERIMENT_NAME = 'sdd_longterm1'
DATASET_NAME = 'sdd'

TRAIN_DATA_PATH = './train.csv'
VAL_DATA_PATH = './val.csv'
TEST_DATA_PATH = './test.csv'
TEST_IMAGE_PATH = 'C:/Users/abcd/YNet/Human-Path-Prediction/data/test/'
TRAIN_IMAGE_PATH = 'C:/Users/abcd/YNet/Human-Path-Prediction/data/train/'  # only needed for YNet, PECNet ignores this value
VAL_IMAGE_PATH = 'C:/Users/abcd/YNet/Human-Path-Prediction/data/val/'
OBS_LEN = 1  # in timesteps(5)
PRED_LEN = 1  # in timesteps(30)
NUM_GOALS = 1  # K_e
NUM_TRAJ = 5  # K_a

BATCH_SIZE = 1

ROUNDS = 3

# model_a = torch.load('./ynet_additional_files/segmentation_models/SDD_segmentation.pth')
# model_a.eval()
# weights = dict()
# for name, para in model_a.named_parameters():
#     weights[name] = para



with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]
params



df_train = pd.read_csv(TRAIN_DATA_PATH)
df_val = pd.read_csv(VAL_DATA_PATH)
df_test = pd.read_csv(TEST_DATA_PATH)

df_train = df_train.drop(df_train[df_train.label == 'handle'].index)
df_train = df_train.drop(df_train[df_train.label == 'middle'].index)
df_val = df_val.drop(df_val[df_val.label == 'handle'].index)
df_val = df_val.drop(df_val[df_val.label == 'middle'].index)
df_test = df_test.drop(df_test[df_test.label == 'handle'].index)
df_test = df_test.drop(df_test[df_test.label == 'middle'].index)

df_train = df_train.drop(['Unnamed: 0','label'],axis=1)
df_val = df_val.drop(['Unnamed: 0','label','filename'],axis=1)

df_test['x'] = (df_test['xmin']+df_test['ymax'])/2
df_test['y'] = (df_test['ymin']+df_test['ymax'])/2
df_test = df_test.drop(['label','xmin','xmax','ymin','ymax'],axis=1)
# # import os
# # print('start checking')
# df_train['filename'] = 'C:/Users/abcd/YNet/Human-Path-Prediction/data/train/' + df_train.sceneId+'/' + df_train.frame

# # file = ['C:/Users/abcd/YNet/Human-Path-Prediction/data/train/'+dirs+'/'+frame for dirs in os.listdir(TRAIN_IMAGE_PATH) for frame in df_train.frame if frame in os.listdir('C:/Users/abcd/YNet/Human-Path-Prediction/data/train/'+dirs+'/')]
# # df_train = df_train[df_train.filename.isin(file)]
# # df_train.to_csv('train.csv')
    
# df_val['filename'] = 'C:/Users/abcd/YNet/Human-Path-Prediction/data/val/' + df_val.sceneId+'/' + df_val.frame

# # file = ['C:/Users/abcd/YNet/Human-Path-Prediction/data/val/'+ dirs +'/' + frame for dirs in os.listdir(VAL_IMAGE_PATH) for frame in df_val.frame if frame in os.listdir('C:/Users/abcd/YNet/Human-Path-Prediction/data/val/'+dirs+'/')]
# # df_val = df_val[df_val.filename.isin(file)]
# # df_val.to_csv('val.csv')

# print('end checking')

model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)
# model.load(f'./ynet_additional_files/pretrained_models/{experiment_name}_weights.pt')
# model.load('./sdd_longterm_weights.pt')
gpu = torch.device('cuda:0')
model.train(df_train, df_val, params, train_image_path=TRAIN_IMAGE_PATH, val_image_path = VAL_IMAGE_PATH
               , experiment_name = EXPERIMENT_NAME, batch_size=BATCH_SIZE,
               num_goals=NUM_GOALS, num_traj=NUM_TRAJ, device=gpu, dataset_name=DATASET_NAME)
experiment_name = EXPERIMENT_NAME
model.load(f'./ynet_additional_files/pretrained_models/{experiment_name}_weights.pt')
model.evaluate(df_test, params, image_path=TEST_IMAGE_PATH,batch_size=BATCH_SIZE, 
               num_goals=NUM_GOALS, num_traj=NUM_TRAJ, device=gpu, dataset_name=DATASET_NAME)
# model.save(f'./result/{experiment_name}_weight_1.pt')