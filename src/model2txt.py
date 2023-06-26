#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import argparse
import torchvision
import torch.nn as nn
from einops import repeat
import torch.optim as optim
from einops.layers.torch import Rearrange
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(add_help=True)

parser.add_argument("--model", type=str, default="../result/data_1/model.pth")

args = parser.parse_args()

if __name__ == "__main__":
    net = torch.load(args.model)
    print(net)
