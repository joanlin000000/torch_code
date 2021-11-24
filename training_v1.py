#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import yaml
import argparse
import torch
from model import YNet


CONFIG_FILE_PATH = './ynet_additional_files/config/sdd_longterm.yaml'  # yaml config file containing all the hyperparameters
EXPERIMENT_NAME = 'sdd_longterm'
DATASET_NAME = 'sdd'

TRAIN_DATA_PATH = './train.csv'
VAL_DATA_PATH = './val.csv'
TRAIN_IMAGE_PATH = 'C:/Users/abcd/YNet/Human-Path-Prediction/data/train/'  # only needed for YNet, PECNet ignores this value
VAL_IMAGE_PATH = 'C:/Users/abcd/YNet/Human-Path-Prediction/data/val/'
OBS_LEN = 5  # in timesteps
PRED_LEN = 30  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a

BATCH_SIZE = 1

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




df_train = df_train.drop(['Unnamed: 0'],axis=1).iloc[:64]
df_val = df_val.drop(['Unnamed: 0'],axis=1).iloc[:64]

# import os
# print('start checking')
df_train['filename'] = 'C:/Users/abcd/YNet/Human-Path-Prediction/data/train/' + df_train.sceneId+'/' + df_train.frame

# file = ['C:/Users/abcd/YNet/Human-Path-Prediction/data/train/'+dirs+'/'+frame for dirs in os.listdir(TRAIN_IMAGE_PATH) for frame in df_train.frame if frame in os.listdir('C:/Users/abcd/YNet/Human-Path-Prediction/data/train/'+dirs+'/')]
# df_train = df_train[df_train.filename.isin(file)]
# df_train.to_csv('train.csv')
    
df_val['filename'] = 'C:/Users/abcd/YNet/Human-Path-Prediction/data/val/' + df_val.sceneId+'/' + df_val.frame

# file = ['C:/Users/abcd/YNet/Human-Path-Prediction/data/val/'+ dirs +'/' + frame for dirs in os.listdir(VAL_IMAGE_PATH) for frame in df_val.frame if frame in os.listdir('C:/Users/abcd/YNet/Human-Path-Prediction/data/val/'+dirs+'/')]
# df_val = df_val[df_val.filename.isin(file)]
# df_val.to_csv('val.csv')

# print('end checking')

model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)

# model.load(f'./ynet_additional_files/pretrained_models/{experiment_name}_weights.pt')
# model.load('./sdd_longterm_weights.pt')

model.train(df_train, df_val, params, train_image_path=TRAIN_IMAGE_PATH, val_image_path = VAL_IMAGE_PATH
               , experiment_name = experiment_name, batch_size=BATCH_SIZE,
               num_goals=NUM_GOALS, num_traj=NUM_TRAJ, device=None, dataset_name=DATASET_NAME)
