import torch
from attunet import AttU_Net as AttU_Net
import data_loader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import cv2

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Torch device: {device}")

LOAD_WEIGHTS = False  
NUM_CLASSES = 55
exp_name = "2025-03-01_14-26-45_AttU_Net"
save_dir = f"/Users/emili/Documents/Cours_2024_2025/data_challenge/models/{exp_name}"
model_path = f"{save_dir}/best_model_state_dict.pt"

data_dir = Path("/Users/emili/Documents/Cours_2024_2025/data_challenge/data")
def load_test_dataset(dataset_dir):
    dataset_dict = {}
    # Note: It's very important to load the images in the correct numerical order!
    for image_file in list(sorted(Path(dataset_dir).glob("*.png"), key=lambda filename: int(filename.name.rstrip(".png")))):
        base_name = image_file.name
        dataset_dict[base_name] = (cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE))
    return dataset_dict
    
model = AttU_Net(img_ch=1, output_ch=NUM_CLASSES)
torch.mps.empty_cache()
model = model.to(device)
print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


model.load_state_dict(torch.load(model_path))
print("Loaded pre-trained weights...")

model.eval()
total_preds = []
columns = []
H, W = 256, 256

# df = pd.DataFrame(np.zeros((H*W, 1)))
lines = ["Pixel " + str(k) for k in range(H*W)]
# df.index = lines
# print(df)
# df.to_csv(f"{save_dir}/predictions.csv", index=True)
# # print(lines)
# ue+=1

for name, img in load_test_dataset(data_dir/"test-images").items():
    columns.append(name)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    img = img.to(device)
    preds = model(img.unsqueeze(0))
    total_preds.append(preds.argmax(1).cpu().numpy())
    # print(preds.shape)
    # print(preds.argmax(1).shape)
    # plt.subplot(1, 2, 1)
    # plt.imshow(img[0].cpu().numpy(), cmap='gray')
    # plt.subplot(1, 2, 2)
    # plt.imshow(preds.argmax(1).cpu().numpy()[0], cmap='gray')
    # plt.show()
    

# predictions est une liste de predictions 2D, soit un array numpy 3D de taille (N, 256, 256)
predictions = np.array(total_preds)
print(predictions.shape)
df = pd.DataFrame(predictions.reshape((predictions.shape[0], -1))).T
# add columns and lines
df.columns = columns
df.index = lines
df.to_csv(f"{save_dir}/predictions.csv", index=True)