#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
from einops import repeat
import torch.optim as optim
from einops.layers.torch import Rearrange
import torchvision.transforms as transforms

sys.path.append('.')

from src.vit_model import *

parser = argparse.ArgumentParser(add_help=True)

parser.add_argument("--model", type=str, default="../result/data_1/model.pth")

args = parser.parse_args()

if __name__ == "__main__":
    net = ViT(
        image_size=32,
        patch_size=4,
        n_classes=10,
        dim=256,
        depth=3,
        n_heads=4,
        mlp_dim = 256
    ).to('cpu')
    print(args.model)
    net.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    print(net)

    arrays = net.layer.weight.cpu().numpy()
    print(arrays)
