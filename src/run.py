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
import os

import data_loader
import losses
import training
from attunet import AttU_Net
from MISSFormer import MISSFormer
import datetime
from torch.utils.tensorboard import SummaryWriter


device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Torch device: {device}")
NETWORK = "AttU_Net" 
# NETWORK = "Missformer"

LOAD_WEIGHTS = False  
NUM_CLASSES = 55
EPOCHS = 60
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if LOAD_WEIGHTS:
    exp_name = "2025-03-01_14-26-45_AttU_Net"
    SAVE_DIR = f"./models/{exp_name}"
    LOG_DIR = f"./logs/{exp_name}"
else:
    exp_name = f"{date}_{NETWORK}"
    SAVE_DIR = f"./models/{exp_name}"
    os.makedirs(SAVE_DIR, exist_ok=True)
    LOG_DIR = f"./logs/{exp_name}"
    os.makedirs(LOG_DIR, exist_ok=True)


def main():
    data_dir = Path("/Users/emili/Documents/Cours_2024_2025/data_challenge/data")

    train_loader, val_loader, X_test = data_loader.data_to_dataloader(data_dir, batch_size=16)
    
    writer = SummaryWriter(LOG_DIR)
    if NETWORK == "Missformer":
        model = MISSFormer(num_classes=NUM_CLASSES, in_ch=1)
    elif NETWORK == "AttU_Net":
        model = AttU_Net(img_ch=1, output_ch=NUM_CLASSES)
    else:
        raise ValueError("Invalid network name")
    torch.mps.empty_cache()
    model = model.to(device)
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    os.makedirs(SAVE_DIR, exist_ok=True)
    model_path = f"{SAVE_DIR}/best_model_state_dict.pt"

    if LOAD_WEIGHTS:
        model.load_state_dict(torch.load(model_path))
        print("Loaded pre-trained weights...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)


    best_model, model, res = training.train(
        model, 
        device, 
        train_loader,
        val_loader,
        
        EPOCHS,
        
        losses.criterion,
        optimizer,
        scheduler,

        save_dir = SAVE_DIR,
        save_file_id = None,

        writer = writer
    )
    


if __name__ == "__main__":
    main()