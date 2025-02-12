import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import cv2
import time
import os
import json

import resnet50
from config import config, update_config


def offline_profiling():
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


if __name__ == '__main__':
    offline_profiling()