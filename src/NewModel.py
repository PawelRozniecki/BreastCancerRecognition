import copy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt  # pip install matplotlib

from PIL import Image
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
from torchvision import models

from matplotlib import pyplot
from torchvision import transforms
import torchvision.transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

NO_WORKERS = 12
EPOCH = 2
DATASET_PATH = '..' + os.path.sep + 'dataset'
TRAINED_MODEL_PATH = '..' + os.path.sep + 'savedModel' + os.path.sep + 'trainedModel.pth'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NewModel(nn.Module) :


    def __init__(self, original_model, num_classes) :
        super(NewModel, self).__init__()
        self.features = original_model.features

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.modelName = 'alexnet'

        # for p in self.features.parameters() :
        #     p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


dataset_path = "/Users/pingwin/Documents/GitHub/AI_Project/dataset"
train_transform = transforms.Compose([
    transforms.Resize(254),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                         )])
validation_split = 0.8
dataset = ImageFolder(dataset_path, transform=train_transform)

# train, test = data.random_split(dataset, [30, 277494])
train, test = data.random_split(dataset, [185016, 92508])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_set = data.DataLoader(train, batch_size=64, shuffle=True, num_workers=2)
# val_set = data.DataLoader(val, batch_size=64, shuffle=True, num_workers=2)
test_set = data.DataLoader(test, batch_size=128, shuffle=False, num_workers=2)

dir(models)
alexnet = models.alexnet(pretrained=True)

net = NewModel(alexnet, 2)

for p in net.features.parameters() :
    p.requires_grad = False

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

best_model_wts = copy.deepcopy(alexnet.state_dict())
best_acc = 0.0

correct = 0
total = 0
running_corrects = 0
for epoch in range(10):

    net.train()
    running_loss = 0.0
    running_corrects = 0
    error = 0.0

    for i, d in enumerate(train_set):

        optimizer.zero_grad()
        print(i, "/", len(train_set))
        data_input, labels = d
        labels = labels.to(device)

        output = net(data_input)
        _, predictions = torch.max(output, 1)
        loss = loss_func(output, labels)
        error = error + loss.item()
        print("loss: ", loss)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data_input.size(0)
        # print(running_loss)
        error = error / len(train)
        running_corrects += torch.sum(predictions == labels)
        epoch_loss = running_loss/len(train)
        epoch_acc = running_corrects.double()/len(train)

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(net.state_dict())
            print("best acc: ", best_acc)

    net.eval()
    for da in test_set :
        data_in, label = da
        o = net(data_in)
        for idx, i in enumerate(o) :
            print(idx, "/", len(test_set))
            if torch.argmax(i) == label[idx] :
                # print("correct: ", correct)
                correct += 1
            total += 1
            print("Error: ", error, "accuracy: ", round(correct / total, 3), "correct: ", correct, " total: ", total)

PATH = '/Users/pingwin/Documents/GitHub/AI_Project/savedModel/PleaseWorkModel.pth'

torch.save(net.state_dict(), PATH)


exit()
