# import pickle
# import os
# handle = open('/home-local2/heweb4.extra.nobkp/datasets/LavalIndoor/pkl/9C4A9102-others-00-1.67716-1.00567.pickle', 'rb')
# root = '/home-local2/heweb4.extra.nobkp/datasets/LavalIndoor/pkl/'
# files = os.listdir(root)
# for f in files:
#     handle = open(root+f, 'rb')
#     p = pickle.load(handle)
#     if 'depth' not in p.keys():
#         print(f)
#         os.system('rm '+root+f)
# exit()
# gt = pickle.load(handle)
# a=1

import os
import os.path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import util
from PIL import Image
import cv2
import pickle
import imageio
imageio.plugins.freeimage.download()
import shutil

def world2latlong(x, y, z):
    """Get the (u, v) coordinates of the point defined by (x, y, z) for
    a latitude-longitude map."""
    u = 1 + (1 / np.pi) * np.arctan2(x, -z)
    v = (1 / np.pi) * np.arccos(y)
    # because we want [0,1] interval
    u = u / 2
    return u, v

def sphere_points(n=128):
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    # print(z)
    radius = np.sqrt(1 - z * z)

    points = np.zeros((n, 3))
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = z

    # xyz = points
    # x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return points

def depth_for_anchors_calc(depth):
    points = sphere_points()
    depth_for_anchors = []
    uvs = world2latlong(points[:,0],points[:,1],points[:,2])
    # pano = np.zeros((128,256,1))
    uvs = np.array(uvs).transpose()
    # for uv in uvs:
    #     pano[int(uv[1]*128),256-int(uv[0]*256)] = 255
    # depth_n = depth/depth.max()
    for uv in uvs:
        depth_for_anchors.append(depth[int(uv[1]*128),256-int(uv[0]*256)])
        # color = (255*depth_n[int(uv[1]*128),256-int(uv[0]*256)], 255*depth_n[int(uv[1]*128),256-int(uv[0]*256)], 255*depth_n[int(uv[1]*128),256-int(uv[0]*256)])
        # cv2.circle(pano,(256-int(uv[0]*256),int(uv[1]*128)),5,color,-1)
    # anchors_indices = np.array(np.nonzero(pano))[0:2,:].transpose()
    # for a in anchors_indices:
    #     depth_for_anchors.append(depth[a[0],a[1]])
    # cv2.imwrite('a.png', pano)
    return np.array(depth_for_anchors)
# handle = open('/home-local2/heweb4.extra.nobkp/datasets/LavalIndoor/pkl/AG8A0008-others-00-2.12407-1.00948.pickle', 'rb')
# gt = pickle.load(handle)
# gt_depth = depth_for_anchors_calc(gt['depth'])
# import util

class ParameterDataset(Dataset):
    def __init__(self, train_dir):
        assert os.path.exists(train_dir)

        # self.train_dir = train_dir
        self.pairs = []

        gt_dir = train_dir + 'pkl/'
        crop_dir = train_dir + 'hdrInputs/'

        gt_nms = os.listdir(gt_dir)
        for nm in gt_nms:
            if nm.endswith('pickle'):
                gt_path = gt_dir + nm
                crop_path = crop_dir + nm.replace('pickle', 'exr')
                if os.path.exists(crop_path):
                    self.pairs.append([crop_path, gt_path])
        # self.pairs = self.pairs[: 1000]
        self.data_len = len(self.pairs)
        self.to_tensor = transforms.ToTensor()
        # self.normalize = transforms.Normalize(mean=[0.48548178, 0.48455666, 0.46329196],
        #                                       std=[0.21904471, 0.21578524, 0.23359051])

        self.tone = util.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
        self.handle = util.PanoramaHandler()

    def __getitem__(self, index):
        training_pair = {
            "crop": None,
            "distribution": None,
            'intensity': None,
            'rgb_ratio': None,
            'ambient': None,
            'depth': None,
            'name': None}

        pair = self.pairs[index]
        crop_path = pair[0]

        exr = self.handle.read_hdr(crop_path)
        input, alpha = self.tone(exr)
        training_pair['crop'] = self.to_tensor(input)

        gt_path = pair[1]
        handle = open(gt_path, 'rb')

        gt = pickle.load(handle)

        gt_depth = depth_for_anchors_calc(gt['depth'])
        
        training_pair['distribution'] = torch.from_numpy(gt['distribution']).float()
        training_pair['intensity'] = torch.from_numpy(np.array(gt['intensity'])).float() * alpha / 500
        training_pair['rgb_ratio'] = torch.from_numpy(gt['rgb_ratio']).float()
        training_pair['ambient'] = torch.from_numpy(gt['ambient']).float() * alpha / (128 * 256)
        training_pair['depth'] = torch.from_numpy(gt_depth).float()

        training_pair['name'] = gt_path.split('/')[-1].split('.pickle')[0]


        # print (training_pair['intensity'], alpha)

        return training_pair

    def __len__(self):
        return self.data_len
