
import copy
from multiprocessing.spawn import freeze_support

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils import data
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from src.constants import *
from src.model import Model


def main() :
    dataset_transform = transforms.Compose([
        transforms.Resize(254),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(0,270),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = ImageFolder(DATASET_PATH, transform=dataset_transform)
    print(len(dataset))
    test_len = int(len(dataset) / 3)
    train_len = int(len(dataset) - test_len)

    train, test = data.random_split(dataset, [train_len, test_len ])

    train_set = data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NO_WORKERS)
    # val_set = data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True, num_workers=NO_WORKERS)
    test_set = data.DataLoader(test, batch_size=len(test), shuffle=False, num_workers=NO_WORKERS)

    alexnet = models.alexnet(pretrained=True)
    model = Model(alexnet, 2)
    model.to(DEVICE)

    for p in model.features.parameters() :
        p.requires_grad = False

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_model_wts = copy.deepcopy(alexnet.state_dict())
    best_acc = 0.0

    correct = 0
    total = 0
    running_corrects = 0
    model.train()
    for epoch in range(EPOCH) :
        error = 0.0
        correct = 0
        total = 0

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
            print("loss: ", loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * image_batch.size(0)
            print(running_loss)
            error = error / len(train)
            running_corrects += torch.sum(predictions.to(DEVICE) == label_batch)
            epoch_loss = running_loss / len(train)
            epoch_acc = running_corrects.double() / len(train)

            if epoch_acc > best_acc :
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print("best acc: ", best_acc)
    model.eval()
    for i, d in enumerate(test_set) :
        print("iteration: ", i, "/", len(test_set))
        test_image, test_label = d
        test_image = test_image.to(DEVICE)
        test_label = test_label.to(DEVICE)

        output = model(test_image)
        _, predicted = torch.max(output.data, 1)
        total += test_label.size(0)
        correct += (predicted == test_label).sum().item()
        print("correct: ", correct, " total: ", total)

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    torch.save(model.state_dict(), TRAINED_MODEL_PATH)




if __name__ == '__main__' :
    freeze_support()
    main()
    exit()
