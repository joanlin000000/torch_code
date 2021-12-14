# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 09:20:35 2021

@author: abcd
"""
#library imports
import os
import random
import math
from datetime import datetime
from collections import Counter
import pandas as pd
import numpy as np

import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
# import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from ipywidgets import IntProgress


# ===========================================================================
# Read csv
# ===========================================================================
# csv_path = 'C:/Users/abcd/bbox_pred/train.csv'
img_path = 'C:/Users/abcd/yolor/dataset/train/images/'
csv_path = 'C:/Users/abcd/bbox_pred/train_1.csv'

df_train = pd.read_csv(csv_path)
print(df_train.shape)
'''
print('===== start: check if file in csv ========')
file = [filename for filename in df_train.filename if filename in os.listdir(img_path)]
df_train = df_train[df_train.filename.isin(file)]
df_train.to_csv('train_1.csv')
df_img_lost_bb = df_train[~df_train.filename.isin(file)]
print('===== end: stop checking file ========')
'''
# df_train.to_csv('train.csv')
# df_img_lost_bb.to_csv('train_imgs_lost_bbox.csv')
df_train['filename'] = img_path + df_train['filename']
df_train.head()
df_train = df_train.drop(['Unnamed: 0'], axis=1)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# im = cv2.imread('{}{}'.format(img_path, df_train['filename'][0]))
# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)
# im /= 255.0
# plt.imshow(im)

#Reading an image
def read_image(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im
def create_mask(bb, x):
    """Creates a mask for the bounding box of same shape as image"""
    rows,cols,*_ = x.shape
    Y = np.zeros((rows, cols))
    bb = bb.astype(np.int)
    Y[bb[0]:bb[2], bb[1]:bb[3]] = 1.
    return Y

def mask_to_bb(Y):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    cols, rows = np.nonzero(Y)
    if len(cols)==0: 
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)

def create_bb_array(x):
    """Generates bounding box array from a train_df row"""
    return np.array([x[5],x[4],x[7],x[6]])

def resize_lost_bb_image(read_path,write_path,sz):
    im = read_image(read_path)
    im_resized = cv2.resize(im, (int(1.49*sz), sz))
    new_path = str(write_path/read_path.split('images/')[-1])
    cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
    return new_path

def resize_image_bb(read_path,write_path,bb,sz):
    """Resize an image and its bounding box and write image to new path"""
    im = read_image(read_path)
    im_resized = cv2.resize(im, (int(1.49*sz), sz))
    Y_resized = cv2.resize(create_mask(bb, im), (int(1.49*sz), sz))
    new_path = write_path + '/' + read_path.split('images/')[-1]
    cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
    return new_path, mask_to_bb(Y_resized)


def resize_test_image_bb(read_path,write_path,bb,sz):
    """Resize an image and its bounding box and write image to new path"""
    im = read_image(read_path)
    im_resized = cv2.resize(im, (int(1.49*sz), sz))
    Y_resized = cv2.resize(create_mask(bb, im), (int(1.49*sz), sz))
    new_path = str(write_path/read_path.split('images/')[-1])
    cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
    return new_path, mask_to_bb(Y_resized)

#Populating Training DF with new paths and bounding boxes
new_paths = []
new_bbs = []
# past_bbs = []
train_path_resized = 'C:/Users/abcd/bbox_pred/images/train_image_300'
print('=== start to resize images ===')
array = df_train.values
for length in range(len(df_train.values)):
    new_path,new_bb = resize_image_bb(str(array[length][0]), train_path_resized,
                                      create_bb_array(array[length]),300)
    # past_bbs.append(create_bb_array(row.values))
    new_paths.append(new_path)
    new_bbs.append(new_bb)
df_train['new_path'] = new_paths
df_train['new_bb'] = new_bbs
print('=== stop resizing ===')

# img_lost_bb_path = './images/img_lost_bb_600(blending)'
# img_lost_bb_new_paths = []
# for _, row in df_img_lost_bb.iterrows():
#     new_path_lost = resize_lost_bb_image(row['filename'], img_lost_bb_path, 600)
#     img_lost_bb_new_paths.append(new_path_lost)
# df_img_lost_bb['new_path'] = img_lost_bb_new_paths


# # Sample Image
# im = cv2.imread(str(df_train.values[58][0]))
# bb = create_bb_array(df_train.values[58])
# print(im.shape)

# Y = create_mask(bb, im)
# mask_to_bb(Y)

# ===========================================================================
# Data Augmentation
# ===========================================================================

# modified from fast.ai
def crop(im, r, c, target_r, target_c): 
    return im[r:r+target_r, c:c+target_c]

# random crop to the original size
def random_crop(x, r_pix=8):
    """ Returns a random crop"""
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    return crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)

def center_crop(x, r_pix=8):
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    return crop(x, r_pix, c_pix, r-2*r_pix, c-2*c_pix)

def rotate_cv(im, deg, y=False, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees"""
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)
    if y:
        return cv2.warpAffine(im, M,(c,r), borderMode=cv2.BORDER_CONSTANT)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def random_cropXY(x, Y, r_pix=8):
    """ Returns a random crop"""
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    xx = crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)
    YY = crop(Y, start_r, start_c, r-2*r_pix, c-2*c_pix)
    return xx, YY

def transformsXY(path, bb, transforms = True): # , transforms
    x = cv2.imread(str(path)).astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
    Y = create_mask(bb, x)
    # if transforms:
    #     rdeg = (np.random.random()-.50)*20
    #     x = rotate_cv(x, rdeg)
    #     Y = rotate_cv(Y, rdeg, y=True)
    #     if np.random.random() > 0.5: 
    #         x = np.fliplr(x).copy()
    #         Y = np.fliplr(Y).copy()
    #     x, Y = random_cropXY(x, Y)
    # else:
    #     x, Y = center_crop(x), center_crop(Y)
    return x, mask_to_bb(Y)

def create_corner_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], color=color,
                         fill=False, lw=3)

def show_corner_bb(im, bb):
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb))
    plt.show()
    

# after transformation
# im, bb = transformsXY(str(df_train.values[0][8]),df_train.values[0][9],True )
# show_corner_bb(im, bb)


# ===========================================================================
# Dataset
# ===========================================================================
df_train = df_train.reset_index()
X = df_train[['new_path', 'new_bb']]
Y = df_train['class']
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

def normalize(im):
    """Normalizes images with Imagenet stats."""
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im - imagenet_stats[0])/imagenet_stats[1]

class Dataset(Dataset):
    def __init__(self, paths, bb, y):
        # super().__init__()
        # self.transforms = transforms
        self.paths = paths.values
        self.bb = bb.values
        self.y = y.values
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        y_class = self.y[idx]
        x, y_bb = transformsXY(path, self.bb[idx])
        # x = normalize(x)
        x = np.rollaxis(x, 2)
        return x, y_class, y_bb

train_ds = Dataset(X_train['new_path'],X_train['new_bb'] ,y_train)
valid_ds = Dataset(X_val['new_path'],X_val['new_bb'],y_val)

batch_size = 128
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, drop_last=True)


class BB_model(nn.Module):
    def __init__(self):
        super(BB_model, self).__init__()
        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())[:5] #:8
        self.features1 = nn.Sequential(*layers[:3]) # :6, first 6 layers of resnet.children
        self.features2 = nn.Sequential(*layers[3:]) # 6:, last 2 layers pf resnet.children
        self.classifier = nn.Sequential(nn.BatchNorm1d(64), nn.Linear(64, 4)) #512
        self.bb = nn.Sequential(nn.BatchNorm1d(64), nn.Linear(64, 4)) #512
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x), self.bb(x)


def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr
        
def val_metrics(model, valid_dl, C=1000):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0 
    # out_bbs=[]
    for x, y_class, y_bb in valid_dl:
        
        batch = y_class.shape[0]
        x = x.float()
        # x = x.cuda().float()
        # y_class = y_class.cuda()
        # y_bb = y_bb.cuda().float()
        y_bb = y_bb.float()
        out_class, out_bb = model(x)
        # out_bbs.append(out_bb)
        loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
        loss_bb = F.mse_loss(out_bb, y_bb, reduction="none").sum(1)
        loss_bb = loss_bb.sum()
        loss = loss_class + loss_bb/C
        _, pred = torch.max(out_class, 1)
        correct += pred.eq(y_class).sum().item()
        sum_loss += loss.item()
        total += batch
    # torch.cuda.empty_cache()    
    return sum_loss/total, correct/total

def train_epocs(model, optimizer, train_dl, val_dl,lr, epochs=10,C=1000):
    train_losses = []
    val_losses = []
    idx = 0
    for i in range(epochs):
        print('epoch:',i+71)
        model.train()
        total = 0
        sum_loss = 0
        for x, y_class, y_bb in train_dl:
            batch = y_class.shape[0]
            x = x.float()
            # x = x.cuda().float()
            # y_class = y_class.cuda()
            # y_bb = y_bb.cuda().float()
            y_bb = y_bb.float()
            out_class, out_bb = model(x)
            
            # show_corner_bb(x, out_bb)
            # model.eval()
            loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
            loss_bb = F.mse_loss(out_bb, y_bb, reduction="none").sum(1)
            loss_bb = loss_bb.sum()
            loss = loss_class + loss_bb/C
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += loss.item()
        torch.cuda.empty_cache()
        train_loss = sum_loss/total
        val_loss, val_acc = val_metrics(model, valid_dl, C)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        PATH = f'./model_{lr}_{i+71}.pth'
        torch.save(model.state_dict(), PATH)
        print('weight saved')
        print("train_loss %.3f val_loss %.3f val_class_acc %.3f" % (train_loss, val_loss, val_acc))
    total_list = pd.DataFrame(
    {'training loss': train_losses,
     'validation loss': val_losses
    })    
    total_list.to_csv(f'result_11_50ep_{lr}.csv')
    return sum_loss/total

lr = 0.003
# for i in range(9):
#     print('lr=',lr)
#     model = BB_model()
#     # model.load_state_dict(torch.load(f'./model_{lr}.pth'))
#     model.eval()
#     parameters = filter(lambda p: p.requires_grad, model.parameters())
#     optimizer = torch.optim.Adam(parameters, lr)
#     train_epocs(model, optimizer, train_dl, valid_dl,lr, epochs=1)
#     lr-=0.002
#     update_optimizer(optimizer, 0.005)
#     train_epocs(model, optimizer, train_dl, valid_dl, epochs=10)
    
#     # save model's weight
#     PATH = f'./model_{lr}.pth'
#     torch.save(model.state_dict(), PATH)
model = BB_model()
model.load_state_dict(torch.load('./weights/model_0.003_83.pth'))
# load_path = 'model_0.009.pth'
# model.load_state_dict(torch.load(load_path))
# model.eval()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters, lr)

train_epocs(model, optimizer, train_dl, valid_dl, lr, epochs=20)
# PATH = f'./model_{lr}_ver4.pth'
# torch.save(model.state_dict(), PATH)
# # update_optimizer(optimizer, 0.005)
# # train_epocs(model, optimizer, train_dl, valid_dl, epochs=10)

# # save model's weight
# PATH = './model_v1.pth'
# torch.save(model.state_dict(), PATH)

# ===========================================================================
# Make Predictions
# ===========================================================================
# test_csv_path = 'C:/Users/abcd/bbox_pred/test.csv'
test_csv_path = 'test1.csv'
test_img_path = 'C:/Users/abcd/yolor/dataset/test/images/'

df_test = pd.read_csv(test_csv_path)
print(df_test.shape)

# file = [filename for filename in df_test.filename if filename in os.listdir(test_img_path)]
# df_test = df_test[df_test.filename.isin(file)]
# df_test.to_csv('test1.csv')
df_test = df_test.drop(['Unnamed: 0'],axis=1)
df_test['filename'] = test_img_path + df_test['filename']
# df_test = df_test.drop(['index'], axis = 1)
df_test.head()

#Populating Training DF with new paths and bounding boxes
new_path_test = []
past_bbs_test = []
new_bbs_test = []
test_path_resized = 'C:/Users/abcd/bbox_pred/images/test_image_300'
# df_test = df_test.drop(['Unnamed: 0'],axis=1)
print('start to resize')
array = df_test.values
for length in range(len(df_test.values)):
    new_path,new_bb = resize_image_bb(str(array[length][0]), test_path_resized, create_bb_array(array[length]),300)
    new_path_test.append(new_path)
    new_bbs_test.append(new_bb)
df_test['new_path'] = new_path_test
df_test['new_bb'] = new_bbs_test
print('stop resizing')

# # Sample Image
# im = cv2.imread(str(df_test.values[58][0]))
# bb = create_bb_array(df_test.values[58])
# print(im.shape)

# Y = create_mask(bb, im)
# mask_to_bb(Y)


X = df_test[['new_path', 'new_bb']]
Y = df_test['class']
batch_size = 1
test_ds = Dataset(paths=X['new_path'], bb=X['new_bb'] , y=Y)
test_dl = DataLoader(test_ds, batch_size=batch_size, drop_last=False,shuffle=False)

def test_metrics(model, test_dl, C=1000):
    global pred_bb
    pred_bb = []
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0 
    with torch.no_grad():
        for x, y_class, y_bb in test_dl:
            batch = y_class.shape[0]
            x = x.float()
            # y_class = y_class.cuda()
            y_bb = y_bb.float()
            out_class, out_bb = model(x)
            
            loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
            loss_bb = F.mse_loss(out_bb, y_bb, reduction="none").sum(1)
            out = out_bb.reshape(4)
            pred_bb.append(out)
            loss_bb = loss_bb.sum()
            loss = loss_class + loss_bb/C
            _, pred = torch.max(out_class, 1)
            correct += pred.eq(y_class).sum().item()
            sum_loss += loss.item()
            total += batch
    return sum_loss/total, correct/total




def test_result(model, test_dl ,C=1000):
    test_loss, test_acc = test_metrics(model, test_dl, C)
    test_loss = test_loss
    test_acc = test_acc
    print("test_loss %.3f test_class_acc %.3f" % (test_loss, test_acc))
    

test_result(model, test_dl)


im_list = X['new_path'].tolist()
test = pd.DataFrame(
    {'new_path': im_list,
     'out_bb': pred_bb,
     'golden_bb': new_bbs_test
    }) 
import matplotlib.patches as patches
for length in range(len(test.values)):
    fig, ax = plt.subplots()
    path, out_bb,golden_bb = test.values[length][0],test.values[length][1].numpy(),test.values[length][2]
    im = cv2.imread(str(path))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = ax.imshow(im)
    patch_out = patches.Rectangle(
        (out_bb[1],out_bb[0]),
        out_bb[3]-out_bb[1],
        out_bb[2]-out_bb[0],
        edgecolor = 'blue',
        facecolor = 'red',
        label='predict_bbox') #xy, size_x,size_y
    patch_golden = patches.Rectangle(
        (golden_bb[1],golden_bb[0]),
        golden_bb[3]-golden_bb[1],
        golden_bb[2]-golden_bb[0],
        edgecolor = 'green',
        facecolor = 'yellow',
        label='golden_bbox')
    ax.add_patch(patch_out)
    ax.add_patch(patch_golden)
    ax.legend([patch_out, patch_golden],['predict_bbox','golden_bbox'])
    plt.show()
    


'''
from GPUtil import showUtilization as gpu_usage
from numba import cuda

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()
    
free_gpu_cache()
'''
# ===========================================================================
# IOU calculation
# ===========================================================================  

# # from sklearn.utils.linear_assignment_ import linear_assignment
# from scipy.optimize import linear_sum_assignment as linear_assignment



# def iou(bb_test, bb_gt):
    
#     # Computes IOU between two bboxes in the form [x1,y1,x2,y2]
#     # Parameters:
#     #     bb_test: [x1,y1,x2,y2,...]
#     #     bb_ground: [x1,y1,x2,y2,...]
#     # Returns:
#     #     score: float, takes values between 0 and 1.
#     #     score = Area(bb_test intersects bb_gt)/Area(bb_test unions bb_gt)
    
#     xx1 = max(bb_test[0], bb_gt[0])
#     yy1 = max(bb_test[1], bb_gt[1])
#     xx2 = min(bb_test[2], bb_gt[2])
#     yy2 = min(bb_test[3], bb_gt[3])
#     w = max(0., xx2 - xx1)
#     h = max(0., yy2 - yy1)
#     area = w * h
#     score = area / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
#                     + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - area)
#     return score


# iou_thresh = 0.6
# def assign(predict_boxes, real_boxes):
#     iou_metric = []
#     for box in predict_boxes:
#         temp_iou = []
#         for box2 in real_boxes:
#             temp_iou.append(iou(box, box2))
#         iou_metric.append(temp_iou)
#     iou_metric = np.array(iou_metric)
#     result = linear_assignment(-iou_metric)
#     output = []
#     output_iou = []
#     for idx in range(len(result)):
#         if iou_metric[result[idx][0],result[idx][1]] > iou_thresh:
#             output.append(result[idx])
#             output_iou.append(iou_metric[result[idx][0],result[idx][1]])
#     return output, output_iou


# #       predict
# #     yes    no
# # yes  TP    FN    real
# # no   FP    TN

# # acc = (TP + TN)/(TP+FN+FP+TN)
# # recall = TP/(TP + FN)
# #
# # 調節score閾值，算出召回率從0到1時的準確率，得到一條曲線
# # 計算曲線的下面積 則為AP

# # ===========================================================================
# # AUC calculation
# # ===========================================================================  

# def get_auc(xy_arr):
#     # 計算曲線下面積即AUC
#     auc = 0.
#     prev_x = 0
#     for x, y in xy_arr:
#         if x != prev_x:
#             auc += (x - prev_x) * y
#             prev_x = x
#     x = [_v[0] for _v in xy_arr]
#     y = [_v[1] for _v in xy_arr]
#     # 畫出auc圖
#     # plt.ylabel("False Positive Rate")
#     # plt.plot(x, y)
#     # plt.show()
#     # print(xy_arr)
#     return auc

# # ===========================================================================
# # AP mAP calculation
# # ===========================================================================  

# def caculate_AP(predict_boxes, real_boxes):
#     recall_arr = []
#     acc_arr = []
#     xy_arr = []
#     score_arr = list(map(lambda i:float(i)*0.01, range(0, 101)))
#     for score in score_arr:
#         temp_predict_boxes = []
#         for box in predict_boxes:
#             if box[4]>score:
#                 temp_predict_boxes.append(box)
#         result,_ = assign(temp_predict_boxes, real_boxes)
#         TP = len(result)
#         FN = len(real_boxes) - TP
#         FP = len(temp_predict_boxes) - TP
#         recall = TP/(TP+FN)
#         acc = TP/(TP+FN+FP)
#         recall_arr.append(recall)
#         acc_arr.append(acc)
#         xy_arr.append([recall,acc])
#     return get_auc(xy_arr)


# def get_mAP(all_predict_boxes, all_real_boxes):
#     ap_arr = []
#     for idx in range(len(all_predict_boxes)):
#         ap_arr.append(caculate_AP(all_predict_boxes[idx], all_real_boxes[idx]))
#     return np.mean(ap_arr)

# get_mAP(pred_bb, new_bbs_test)
