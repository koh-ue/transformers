#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: FIXME: XXX: HACK: NOTE: INTENT: USAGE:

from __future__ import print_function, division

import os
import sys
import copy
import time
import torch
import shutil
import argparse
import numpy as np
import torchvision
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms

sys.path.append(".")

from src.utils import decorators
from src.utils.printcolor import BOLD_BLUE, BOLD_GREEN, BOLD_RED, BOLD_YELLOW, END

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device = {device}")

parser = argparse.ArgumentParser(add_help=True)

parser.add_argument("--zipfile", type=str, default="../result/data_1/UNITV_Training.zip")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=0.1)

args = parser.parse_args()

def check_valid(path):
    path = Path(path)
    return not path.stem.startswith('._')

@decorators.show_start_end
def create_dataloarder(zipfile_path):
    base_dir = os.path.dirname(zipfile_path)
    shutil.unpack_archive(zipfile_path, base_dir)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'vaild': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = '../result/data_1'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x], is_valid_file=check_valid)
                    for x in ['train', 'vaild']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'vaild']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'vaild']}
    class_names = image_datasets['train'].classes

    print('-'*10, 'image_datasets','-'*10,'\n', image_datasets)
    print()
    print('-'*10,'train dataset','-'*10,'\n', image_datasets['train'])
    print()
    print('-'*10,'label','-'*10,'\n', class_names)

    return dataloaders, dataset_sizes, class_names

def tensor_to_np(inp):
  "imshow for Tensor"
  inp = inp.numpy().transpose((1,2,0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1)
  return inp

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = fig.add_subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}  label: {}'
                             .format(class_names[preds[j]], class_names[labels[j]]))
                ax.imshow(tensor_to_np(inputs.cpu().data[j]))

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

if __name__ == "__main__":
    dataloaders, dataset_sizes, class_names = create_dataloarder(args.zipfile)

    model_ft = models.resnet18(weights="DEFAULT")
    print(model_ft)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.5, weight_decay=0.005)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(BOLD_YELLOW + f'Epoch {epoch}/{num_epochs - 1}' + END)
            print(BOLD_YELLOW + '-' * 10 + END)

            # Each epoch has a training and validation phase
            for phase in ['train', 'vaild']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                
                if phase == 'train':
                    scheduler.step()

                # deep copy the model
                if phase == 'vaild' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    # training
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

    torch.save(model_ft.to('cpu').state_dict(), f"{os.path.dirname(args.zipfile)}/model.pth")


