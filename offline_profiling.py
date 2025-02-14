import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
import cv2
import time
import os
import json

import resnet50
import lib.models.faster_voxelpose as fastervoxelpose
from config import config, update_config


def get_cam():
    cameras = dict()
    M = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0],
                  [0.0, 1.0, 0.0]])
    cam_list = [(0, 3), (0, 6), (0, 12), (0, 13), (0, 23)]
    sequence_list = ["custom"]
    for seq in sequence_list:
        cameras[seq] = []

        cam_file = os.path.join(".", "", "calibration_{:s}.json".format(seq))
        with open(cam_file, "r") as f:
            calib = json.load(f)

        for cam in calib["cameras"]:
            if (cam['panel'], cam['node']) in cam_list:
                sel_cam = {}
                sel_cam['K'] = np.array(cam['K'])
                sel_cam['distCoef'] = np.array(cam['distCoef'])
                sel_cam['R'] = np.array(cam['R']).dot(M)
                sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
                cameras[seq].append(sel_cam)

        # convert the format of camera parameters
        for k, v in enumerate(cameras[seq]):
            our_cam = dict()
            our_cam['R'] = v['R']
            our_cam['T'] = -np.dot(v['R'].T, v['t']) * 10.0  # the order to handle rotation and translation is reversed
            our_cam['fx'] = v['K'][0, 0]
            our_cam['fy'] = v['K'][1, 1]
            our_cam['cx'] = v['K'][0, 2]
            our_cam['cy'] = v['K'][1, 2]
            our_cam['k'] = v['distCoef'][[0, 1, 4]].reshape(3, 1)
            our_cam['p'] = v['distCoef'][[2, 3]].reshape(2, 1)
            cameras[seq][k] = our_cam
    return cameras

def offline_profiling_2d():
    update_config("config.yaml")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Read image, convert BGR -> RGB
    image = cv2.imread('dataset/Panoptic/00_03_00000173.jpg', cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # Convert to tensor
    image_tensor = transform(image).float()
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze_(0)

    # load resnet model and weights
    model = resnet50.get(config)
    model.load_state_dict(torch.load("pose_resnet50_panoptic.pth.tar"))

    offline_profile = {"2D_profile": {f"Partition {i}":{} for i in range(21)}}

    trials = 1
    for i in range(21):
        avg = 0
        for n in range (trials):
            start = time.time()
            y = model(image_tensor)
            end = time.time()
            avg += end - start
        offline_profile["2D_profile"][f"Partition {i}"] = avg / trials

    print(offline_profile)

    if not os.path.exists('Profiles/'):
        os.makedirs('Profiles/')


def offline_profiling_3d():
    update_config("config.yaml")

    # Generate random values for 2D backbone
    random_heatmap = torch.rand(size=(1, 5, 15, 128, 240))

    # Define custom Params
    meta = {'seq': ['custom']}
    cameras = get_cam()
    print(f'Camera keys: {cameras.keys()}')
    print(f'Meta seq: {meta["seq"][0]}')
    # load faster-voxelpose model and shape
    model = fastervoxelpose.get(config)


    # Prediction
    y = model(input_heatmaps=random_heatmap,
              meta=meta,
              cameras=cameras)

if __name__ == '__main__':
    offline_profiling_3d()