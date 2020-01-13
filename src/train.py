#!/usr/bin/env python3
import copy
from multiprocessing.spawn import freeze_support
import torch
import torch.nn as nn
import requests
import torch.optim as optim
import torchvision.models as models
from torch.utils import data
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from src.constants import *
import numpy as np
from src.model import Model


def main() :

    alexnet = models.alexnet(pretrained=True)
    model = Model(alexnet, 2)
    model.to(DEVICE)

    dataset_transform = transforms.Compose([
        transforms.Resize(254),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(0,270),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = ImageFolder(DATASET_PATH, transform=dataset_transform)
    test_len = int(len(dataset) / 3)
    train_len = int(len(dataset) - test_len)

    train, test = data.random_split(dataset, [train_len, test_len ])
    train_set = data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NO_WORKERS)
    test_set = data.DataLoader(test, batch_size= 512, shuffle=False, num_workers=NO_WORKERS)


    for p in model.features.parameters() :
        p.requires_grad = False

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # fining the best learning rate
    # lr_finder = LRFinder(model, optimizer, loss_func, device=DEVICE)
    # lr_finder.range_test(train_set, end_lr=100, num_iter=100)
    # lr_finder.plot(log_lr=False)  # to inspect the loss-learning rate graph
    # lr_finder.reset()

    best_model_wts = copy.deepcopy(alexnet.state_dict())
    best_acc = 0.0
    epoch_no_improve = 0

    min_val_loss = np.Inf

#main loop
    for epoch in range(EPOCH) :
        model.train()

        error = 0.0
        correct = 0
        correct_train = 0
        total_train = 0
        total = 0
        val_loss = 0.0

        for i, batch in enumerate(train_set) :

            optimizer.zero_grad()

            running_loss = 0.0
            running_corrects = 0
            print(i + 1, "/", len(train_set))

            image_batch, label_batch = batch
            image_batch = image_batch.to(DEVICE)
            label_batch = label_batch.to(DEVICE)

            predicted_labels = model(image_batch)
            predicted_labels = predicted_labels.to(DEVICE)
            _, predictions = torch.max(predicted_labels, 1)

            loss = loss_func(predicted_labels, label_batch)
            error = error + loss.item()
            # print("loss: ", loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * image_batch.size(0)
            # print(running_loss)
            error = error / len(train)
            running_corrects += torch.sum(predictions.to(DEVICE) == label_batch)
            epoch_loss = np.round(running_loss, 3) / len(train)

            total_train += label_batch.size(0)
            correct_train += (predictions == label_batch).sum().item()
            print("total_train: " , total_train , "total correct: ", correct_train)
            epoch_acc = running_corrects.double() / len(train)

            if epoch_acc > best_acc :
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print("best acc: ", best_acc)

        print('Accuracy of training on the all the images : %d %%' % (
                100 * correct_train / total_train))


        model.eval()
        for i, d in enumerate(test_set) :
            print("iteration: ", i, "/", len(test_set))
            test_image, test_label = d
            test_image = test_image.to(DEVICE)
            test_label = test_label.to(DEVICE)

            output = model(test_image)
            loss = loss_func(output, test_label)
            
            val_loss += loss
            _, predicted = torch.max(output.data, 1)
            total += test_label.size(0)
            correct += (predicted == test_label).sum().item()
            print("correct: ", correct, " total: ", total)

        print('Accuracy of the network on the all the images test images: %d %%' % (
                100 * correct / total))

    # torch.save(model.state_dict(), TRAINED_MODEL_PATH)

    avg_train_loss = loss/len(train_set)
    avg_test_loss = val_loss/len(test_set)

    if avg_test_loss < min_val_loss:
        torch.save(model.state_dict(), TRAINED_MODEL_PATH)
        epoch_no_improve = 0
        min_val_loss = avg_test_loss

    else:
        epoch_no_improve += 1
        if epoch_no_improve == EPOCH:
            print('Early stopping')




    print("avg train: ", avg_train_loss, "avg test: ", avg_test_loss)
if __name__ == '__main__' :
    freeze_support()
    main()
    exit()
