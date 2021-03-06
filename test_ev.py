# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 14:28:26 2021

@author: abcd
"""
import torch
import torch.nn as nn
from util.image_utils import get_patch, image2world

def tests(model, test_loader, test_images, e, obs_len, pred_len, batch_size, params, gt_template, device,epoch, input_template, optimizer, criterion, dataset_name, homo_mat):
    """
	Run training for one epoch

	:param model: torch model
	:param train_loader: torch dataloader
	:param train_images: dict with keys: scene_name value: preprocessed image as torch.Tensor
	:param e: epoch number
	:param params: dict of hyperparameters
	:param gt_template:  precalculated Gaussian heatmap template as torch.Tensor
	:return: train_ADE, train_FDE, train_loss for one epoch
	"""
    test_loss = 0
    test_ADE = []
    test_FDE = []
    # model.train()
    counter = 0
    
	# outer loop, for loop over each scene as scenes have different image size and to calculate segmentation only once
    for batch, (trajectory, meta, scene) in enumerate(test_loader):
		# Stop training after 25 batches to increase evaluation frequency
        if dataset_name == 'sdd' and obs_len == 8 and batch > 25:
            break

		# TODO Delete
        if dataset_name == 'eth':
            print(counter)
            counter += batch_size
			# Break after certain number of batches to approximate evaluation, else one epoch takes really long
        if counter > 30: #TODO Delete
            break


		# Get scene image and apply semantic segmentation
        if e < params['unfreeze']:  # before unfreeze only need to do semantic segmentation once
            model.eval()
            scene_image = test_images[scene].to(device).unsqueeze(0)
            scene_image = model.segmentation(scene_image)
            # model.train()

		# inner loop, for each trajectory in the scene
        final_x = []
        final_y = []
        pred_x = []
        pred_y = []
        for i in range(0, len(trajectory), batch_size):
            if e >= params['unfreeze']:
                scene_image = test_images[scene].to(device).unsqueeze(0)
                scene_image = model.segmentation(scene_image)

			# Create Heatmaps for past and ground-truth future trajectories
            _, _, H, W = scene_image.shape  # image shape

            observed = trajectory[i:i+batch_size, :obs_len, :].reshape(-1, 2).cpu().numpy() # .cpu()
            observed_map = get_patch(input_template, observed, H, W)
            observed_map = torch.stack(observed_map).reshape([-1, obs_len, H, W])

            gt_future = trajectory[i:i + batch_size, obs_len:].to(device)
            gt_future_map = get_patch(gt_template, gt_future.reshape(-1, 2).cpu().numpy(), H, W) # .cpu()
            gt_future_map = torch.stack(gt_future_map).reshape([-1, pred_len, H, W])

            gt_waypoints = gt_future[:, 0] # params['waypoints']
            gt_waypoint_map = get_patch(input_template, gt_waypoints.reshape(-1, 2).cpu().numpy(), H, W) # .cpu()
            gt_waypoint_map = torch.stack(gt_waypoint_map).reshape([-1, 1, H, W]) # gt_waypoints.shape[1],

			# Concatenate heatmap and semantic map
            semantic_map = scene_image.expand(observed_map.shape[0], -1, -1, -1)  # expand to match heatmap size
            feature_input = torch.cat([semantic_map, observed_map], dim=1)

			# Forward pass
			# Calculate features
            features = model.pred_features(feature_input)

			# Predict goal and waypoint probability distribution
            pred_goal_map = model.pred_goal(features)
            goal_loss = criterion(pred_goal_map, gt_future_map) * params['loss_scale']  # BCEWithLogitsLoss

			# Prepare (downsample) ground-truth goal and trajectory heatmap representation for conditioning trajectory decoder
            gt_waypoints_maps_downsampled = [nn.AvgPool2d(kernel_size=2**i, stride=2**i)(gt_waypoint_map) for i in range(1, len(features))]
            gt_waypoints_maps_downsampled = [gt_waypoint_map] + gt_waypoints_maps_downsampled

			# Predict trajectory distribution conditioned on goal and waypoints
            traj_input = [torch.cat([feature, goal], dim=1) for feature, goal in zip(features, gt_waypoints_maps_downsampled)]
            pred_traj_map = model.pred_traj(traj_input)
            traj_loss = criterion(pred_traj_map, gt_future_map) * params['loss_scale']  # BCEWithLogitsLoss

			

            with torch.no_grad():
				
				# Evaluate using Softargmax, not a very exact evaluation but a lot faster than full prediction
                pred_traj = model.softargmax(pred_traj_map)
                pred_goal = model.softargmax(pred_goal_map[:, -1:])
                import cv2
                import matplotlib.pyplot as plt
                import numpy as np
                image = np.array(test_images[scene].permute(1, 2, 0))[:, :, [0, 0, 0]] # 210 120 021 012 201 102 [:, :, [0, 2, 1]]
                # cv2.imshow('My Image', image)
                plt.imshow(image)
                tr = pred_traj.tolist()
                gt = gt_future.tolist()
                gt_rev = gt_future[:, -1:].tolist()
                tr_rev = pred_goal[:, -1:].tolist()
                c = 1
                # print(gt)
                plt.plot(gt[0][0][0],gt[0][0][1],'bs', label='future_traj', linewidth=3)
                plt.plot(tr[0][0][0],tr[0][0][1], 'rx', label = 'pred_traj', linewidth=3)
                plt.plot(gt_rev[0][0][0],gt_rev[0][0][1],'go', label = 'final', linewidth=3)
                plt.plot(tr_rev[0][0][0],tr_rev[0][0][1], 'ys', label = 'pred_final', linewidth=3)
                # plt.savefig(f'./result/img/{c}{epoch}.jpg')
                plt.show()
                final_x.append(gt_rev[0][0][0])
                final_y.append(gt_rev[0][0][1])
                pred_x.append(tr_rev[0][0][0])
                pred_y.append(tr_rev[0][0][1])
                c += 1
				# converts ETH/UCY pixel coordinates back into world-coordinates
				# if dataset_name == 'eth':
				# 	pred_goal = image2world(pred_goal, scene, homo_mat, params)
				# 	pred_traj = image2world(pred_traj, scene, homo_mat, params)
				# 	gt_future = image2world(gt_future, scene, homo_mat, params)

                test_ADE.append(((((gt_future - pred_traj) / params['resize']) ** 2).sum(dim=2) ** 0.5).mean(dim=1))
                test_FDE.append(((((gt_future[:, -1:] - pred_goal[:, -1:]) / params['resize']) ** 2).sum(dim=2) ** 0.5).mean(dim=1))
        import pandas as pd
        test = pd.DataFrame({'final_x': final_x,
                             'final_y': final_y,
                             'pred_x':pred_x,
                             'pred_y':pred_y}) 
        test.to_csv('./result/test_results_v1.csv')
    test_ADE = torch.cat(test_ADE).mean()
    test_FDE = torch.cat(test_FDE).mean()

    return test_ADE, test_FDE, test_loss
