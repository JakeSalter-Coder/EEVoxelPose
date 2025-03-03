import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
import cv2
import time
import os
import json
import pickle
from tqdm import tqdm

import resnet50
import lib.models.faster_voxelpose as fastervoxelpose
from config import config, update_config


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return np.array(b) + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if isinstance(scale, torch.Tensor):
        scale = np.array(scale.cpu())
    if isinstance(center, torch.Tensor):
        center = np.array(center.cpu())
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w, src_h = scale_tmp[0], scale_tmp[1]
    dst_w, dst_h = output_size[0], output_size[1]

    rot_rad = np.pi * rot / 180
    if src_w >= src_h:
        src_dir = get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)
    else:
        src_dir = get_dir([src_h * -0.5, 0], rot_rad)
        dst_dir = np.array([dst_h * -0.5, 0], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift     # x,y
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_scale(image_size, resized_size):
    w, h = image_size
    w_resized, h_resized = resized_size
    if w / w_resized < h / h_resized:
        w_pad = h / h_resized * w_resized
        h_pad = h
    else:
        w_pad = w
        h_pad = w / w_resized * h_resized
    scale = np.array([w_pad / 200.0, h_pad / 200.0], dtype=np.float32)

    return scale


def get_resize_transform():
    ori_image_size = np.array([1920, 1080])
    image_size = np.array([960, 512])
    r = 0
    c = np.array([ori_image_size[0] / 2.0, ori_image_size[1] / 2.0])
    s = get_scale((ori_image_size[0], ori_image_size[1]), image_size)
    trans = get_affine_transform(c, s, r, image_size)
    return trans


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

def offline_profiling_2d(trials=1):
    random_image_tensor = torch.rand(size=(1, 3, 960, 512))

    # load resnet model and weights
    model = resnet50.get(config)
    model.load_state_dict(torch.load("pose_resnet50_panoptic.pth.tar"))

    offline_profile = {"2D_profile_server": {f"Partition {i}":{} for i in range(21)},
                       "2D_profile_client": {f"Partition {i}":{} for i in range(21)}}

    for i in tqdm(range(21)):
        client_avg = 0
        server_avg = 0
        for n in range(trials):
            #client model
            start = time.time()
            client_output = model(x=random_image_tensor, server=False, partition=i)
            end = time.time()
            client_avg += end - start
            #server model
            start = time.time()
            y = model(x=client_output, server=True, partition=i)
            end = time.time()
            server_avg += end - start


        offline_profile["2D_profile_client"][f"Partition {i}"] = client_avg / trials
        offline_profile["2D_profile_server"][f"Partition {i}"] = server_avg / trials

    if not os.path.exists('Profiles/'):
        os.makedirs('Profiles/')

    file = f"Profiles/2D_profile.json"

    with open(file, "w") as f:
        json.dump(offline_profile, f, indent=4)

    print(f"Successfully wrote logs for 2D_profile")


def offline_profiling_3d(trials=1):
    # Generate random values for 2D backbone
    random_heatmap = torch.rand(size=(1, 5, 15, 128, 240))

    # Define custom Params
    meta = {'seq': ['custom']}
    cameras = get_cam()
    resize_transform = torch.as_tensor(get_resize_transform(), dtype=torch.float, device=config.DEVICE)

    # load faster-voxelpose model
    model = fastervoxelpose.get(config)
    model.eval()
    model.load_state_dict(torch.load("model_best.pth.tar"))

    # Timed prediction
    average = 0
    for n in tqdm(range(trials)):
        start = time.time()
        y = model(input_heatmaps=random_heatmap,
                  meta=meta,
                  cameras=cameras,
                  resize_transform=resize_transform)
        end = time.time()
        average += end - start
    offline_profile = {"3D_Profile": average / trials}

    if not os.path.exists('Profiles/'):
        os.makedirs('Profiles/')

    file = f"Profiles/3D_profile.json"

    with open(file, "w") as f:
        json.dump(offline_profile, f, indent=4)

    print("Successfully wrote logs for 3D_profile")


def get_intermediate_output():
    random_image_tensor = torch.rand(size=(1, 3, 960, 512))

    # load resnet model and weights
    model = resnet50.get(config)
    model.load_state_dict(torch.load("pose_resnet50_panoptic.pth.tar"))

    intermediate_output_size = {}
    for i in range(21):
        y = model(x=random_image_tensor, server=False, partition=i)
        y = pickle.dumps(y)
        intermediate_output_size[f"Partition {i}"] = len(y)

    file = "Profiles/intermediate_output.json"
    with open(file, "w") as f:
        json.dump(intermediate_output_size, f, indent=4)

    print("Successfully wrote intermediate output")


if __name__ == '__main__':
    update_config("config.yaml")
    offline_profiling_2d( trials=5)
    #offline_profiling_3d(trials=20)
    #get_intermediate_output()