import numpy as np
from skimage.segmentation import watershed, felzenszwalb
from skimage.filters import sobel
import pandas as pd
from pathlib import Path
import cv2
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.filters import rank
from scipy import ndimage as ndi
from skimage.morphology import disk
import sklearn.metrics
import torch

NUM_CLASSES = 55

def load_dataset(dataset_dir):
    dataset_list = []
    # Note: It's very important to load the images in the correct numerical order!
    for image_file in list(sorted(Path(dataset_dir).glob("*.png"), key=lambda filename: int(filename.name.rstrip(".png")))):
        dataset_list.append(cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE))
    return np.stack(dataset_list, axis=0)

def data_to_dataloader(dataset_dir, batch_size=32):
    X_train = load_dataset(dataset_dir/"train-images")
    Y_train = pd.read_csv(f"{dataset_dir}/y_train.csv", index_col=0).T
    Y_train = torch.tensor(Y_train.values, dtype=torch.int64)
    Y_train = Y_train.reshape(-1, 256, 256)
    # one hot
    # Y_train = torch.nn.functional.one_hot(Y_train, num_classes=NUM_CLASSES)
    X_test = load_dataset(dataset_dir/"test-images")

    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)[:800], Y_train[:800])
    #split the dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

