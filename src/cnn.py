#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CNN
https://tkstock.site/2022/05/29/python-pytorch-mydataset-dataloader/
https://zenn.dev/a5chin/articles/original_data
"""

# TODO: FIXME: XXX: HACK: NOTE: INTENT: USAGE:

import os
import sys
import torch
import shutil
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

sys.path.append(".")

from src.utils import decorators
from src.utils.printcolor import BOLD_BLUE, BOLD_GREEN, BOLD_RED, BOLD_YELLOW, END

parser = argparse.ArgumentParser(add_help=True)

parser.add_argument("--zipfile", type=str, default="../result/data_1/UNITV_Training.zip")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=0.1)

args = parser.parse_args()

BATCH_SIZE = args.batch_size
SIZE = args.size
NUM_EPOCHS = args.num_epochs
# NOTE: AREA for functions.

@decorators.show_start_end
def load_zip(zipfile_path):
    base_dir = os.path.dirname(zipfile_path)
    shutil.unpack_archive(zipfile_path, base_dir)

    try:
        train_dir = os.path.join(base_dir, "train")
        valid_dir = os.path.join(base_dir, "vaild")
        df = pd.DataFrame(columns=['path', 'filename', 'category', 'label'])
        for target_dir in [train_dir, valid_dir]:
            for numbered_class in [file for file in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, file))]:
                for img_file in os.listdir(os.path.join(target_dir, f"{numbered_class}")):
                    if not img_file.startswith('.'):
                        img_df = pd.DataFrame([[f"{target_dir}/{numbered_class}/{img_file}", img_file, int(numbered_class), int(numbered_class)-1]], columns=['path', 'filename', 'category', 'label'])
                        df = pd.concat([df, img_df], ignore_index=True, axis=0)
        return df
    except:
        print(BOLD_RED + 'error: something is wrong.' + END)



class DeviceDataLoader(DataLoader):
    def __init__(self, dl, **kwargs):
        super().__init__(dl.dataset, **kwargs)
        self.dl = dl
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"device = {self.device}")
    
    def __iter__(self):
        for b in self.dl:
            yield self.to_device(b, self.device)
    
    def __len__(self):
        return len(self.dl)
    
    def to_device(self, data, device):
        if isinstance(data, (list, tuple)):
            return [self.to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)
        

class OwnDataset(Dataset):
    def __init__(self, train_df, input_size, phase='train',transform=None):
        super().__init__()
        self.train_df = train_df
        image_paths = train_df["path"].to_list()
        self.input_size = input_size
        self.len = len(image_paths)
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image_path = self.train_df["path"].to_list()[index]
        
        image = Image.open(image_path)
        image = image.resize((300, 300))
        image = np.array(image).astype(np.float32).transpose(2, 1, 0) # Dataloader で使うために転置する
        label = self.train_df["label"].apply(lambda x : int(x)).to_list()[index]
        return image, label

class CNN(nn.Module):
    def __init__(self, out_features_size=4):
        
        super(CNN, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 75 * 75, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=out_features_size)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = x.view(-1, 16 * 75 * 75)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


if __name__ == '__main__':
    data_df = load_zip(args.zipfile)

    image_dataset = OwnDataset(data_df, (SIZE, SIZE))
    train_dataset, valid_dataset = torch.utils.data.random_split( image_dataset, [int(len(image_dataset))-20, 20])

    assert len(train_dataset)%BATCH_SIZE == 0 and len(valid_dataset)%BATCH_SIZE == 0
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    train_dataloader = DeviceDataLoader(train_dataloader, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    valid_dataloader = DeviceDataLoader(valid_dataloader, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    dataloaders_dict = {
        'train': train_dataloader,
        'valid': valid_dataloader
    }

    batch_iterator = iter(dataloaders_dict['train'])
    
    net = CNN(out_features_size=4)
    if torch.cuda.is_available():
        net.cuda('cuda')
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate)
    nll_loss = nn.NLLLoss()
    

    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []

    for epoch in range(NUM_EPOCHS):
        print(BOLD_YELLOW + f'Epoch {epoch+1}/{NUM_EPOCHS}' + END)
        print(BOLD_YELLOW + '-------------' + END)
        for phase in ['train', 'valid']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs, labels in dataloaders_dict[phase]:
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                valid_losses.append(epoch_loss)
                valid_accs.append(epoch_acc.item())
            # if epoch == 0:
            #     model_img = make_dot(outputs, params=dict(net.named_parameters()))
            #     model_img.format = 'png'
            #     model_img.render('graph')

    fig = plt.figure(figsize=(10, 6))
    ax_loss = fig.add_subplot(121)
    ax_loss.plot(np.arange(NUM_EPOCHS), np.array(train_losses))
    ax_loss.grid()
    ax_loss.set_title("Loss per Epoch")

    ax_acc = fig.add_subplot(122)
    ax_acc.plot(np.arange(NUM_EPOCHS), np.array(train_accs))
    ax_acc.grid()
    ax_acc.set_title("Accuracy per Epoch")

    fig.suptitle("Training Data")
    fig.savefig(f"{os.path.dirname(args.zipfile)}/train_loss_accs.pdf")

    fig = plt.figure(figsize=(10, 6))
    ax_loss = fig.add_subplot(121)
    ax_loss.plot(np.arange(NUM_EPOCHS), np.array(valid_losses))
    ax_loss.grid()
    ax_loss.set_title("Loss per Epoch")

    ax_acc = fig.add_subplot(122)
    ax_acc.plot(np.arange(NUM_EPOCHS), np.array(valid_accs))
    ax_acc.grid()
    ax_acc.set_title("Accuracy per Epoch")

    fig.suptitle("Validation Data")
    fig.savefig(f"{os.path.dirname(args.zipfile)}/valid_loss_accs.pdf")
    

