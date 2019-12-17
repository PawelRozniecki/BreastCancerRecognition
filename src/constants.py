import os

from torch import cuda, device

NO_WORKERS = 12
BATCH_SIZE = 256
EPOCH = 2
DATASET_PATH = '..' + os.path.sep + 'dataset'
TRAINED_MODEL_PATH = '..' + os.path.sep + 'trainedModel.pth'
DEVICE = device("cuda:0" if cuda.is_available() else "cpu")
