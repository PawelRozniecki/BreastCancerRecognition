import os

from torch import cuda, device

NO_WORKERS = 12
BATCH_SIZE = 64
EPOCH = 5
DATASET_PATH = '..' + os.path.sep + 'dataset'
TRAINED_MODEL_PATH = '..' + os.path.sep + 'trainedModel3.pth'
LABELS_PATH = '..' + os.path.sep + 'labels.txt'
DEVICE = device("cuda:0" if cuda.is_available() else "cpu")
