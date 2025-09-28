import torch
import numpy as np
from torch.nn import functional as F
from torch.nn import BCELoss, CrossEntropyLoss
import pandas as pd

NUM_CLASSES = 55

EPSILON = 1e-6

class DiceLoss(torch.nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, pred, mask):
        pred = pred.flatten()
        mask = torch.nn.functional.one_hot(mask, num_classes=NUM_CLASSES).permute(0, 3, 1, 2)
        mask = mask.flatten()
        
        intersect = (mask * pred).sum()
        dice_score = 2*intersect / (pred.sum() + mask.sum() + EPSILON)
        dice_loss = 1 - dice_score
        return dice_loss

    
class DiceLossWithLogtis(torch.nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, pred, mask):
        prob = F.softmax(pred, dim=1)
        mask = torch.nn.functional.one_hot(mask, num_classes=NUM_CLASSES).permute(0, 3, 1, 2)
        true_1_hot = mask.to(dtype=prob.dtype)
        # print("pred shape", prob.shape)
        # print("mask shape", true_1_hot.shape)
        
        dims = (0,) + tuple(range(2, true_1_hot.ndimension()))
        intersection = torch.sum(prob * true_1_hot, dims)
        cardinality = torch.sum(prob + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + EPSILON)).mean()
        return (1 - dice_loss)


# def dice_image(prediction, ground_truth):
#     """Compute the dice score for each class (1, 2, ..., NUM_CLASSES)"""
#     intersection = torch.sum(prediction * ground_truth)
    
#     if torch.sum(prediction) == 0 and torch.sum(ground_truth) == 0:
#         return float('nan')
    
#     return 2 * intersection / (torch.sum(prediction) + torch.sum(ground_truth))

# def dice_multiclass(prediction, ground_truth):
#     dices = []
#     for i in range(1, NUM_CLASSES + 1):  # Skip background class (usually 0)
#         dices.append(dice_image(prediction == i, ground_truth == i))
#     return torch.tensor(dices)

# def dice_pytorch(y_true_tensor, y_pred_tensor, num_classes=NUM_CLASSES):
#     """Compute the dice score for each sample in the batch and then average it"""
#     batch_size, height, width = y_true_tensor.shape
#     y_true_tensor = y_true_tensor.view(batch_size, height*width)
#     y_pred_tensor = y_pred_tensor.view(batch_size, height*width)
    
#     y_true_tensor = y_true_tensor.T
#     print("y true shape", y_true_tensor.shape)
#     y_pred_tensor = y_pred_tensor.T
#     print("y pred shape", y_pred_tensor.shape)
    

#     individual_dice = []
    
#     for i in range(height*width):
#         y_true = y_true_tensor[i]  # Aplatir les données de ground truth en un vecteur
#         y_pred = y_pred_tensor[i]  # Aplatir les prédictions one-hot en un vecteur
        
#         dice = dice_multiclass(y_true, y_pred)  # Calcul du Dice pour chaque image
#         individual_dice.append(dice)
#         if i % 100 == 0:
#             print(f"Processed {i} pixels")
    
#     final = torch.stack(individual_dice)  # Garder tout en tensor
#     cls_dices = torch.nanmean(final, axis=0)  # Moyenne par classe
    
#     return torch.nanmean(cls_dices)  # Moyenne sur toutes les classes


# # def criterion(preds, masks):
# #     return dice_pytorch(masks, preds)

def dice_image(prediction, ground_truth):
    intersection = np.sum(prediction * ground_truth)
    if np.sum(prediction) == 0 and np.sum(ground_truth) == 0:
        return np.nan
    return 2 * intersection / (np.sum(prediction) + np.sum(ground_truth))


def dice_multiclass(prediction, ground_truth):
    dices = []
    for i in range(1, NUM_CLASSES + 1): # skip background
        dices.append(dice_image(prediction == i, ground_truth == i))
    return np.array(dices)

def dice_pandas(y_true, y_pred) -> float:
    batch_size, height, width = y_true.shape
    y_true_tensor = y_true.view(batch_size, height*width)
    y_pred_tensor = y_pred.view(batch_size, height*width)
    
    # transform to pandas dataframe
    y_true_df = pd.DataFrame(y_true_tensor.cpu().numpy())
    y_pred_df = pd.DataFrame(y_pred_tensor.cpu().numpy())
    """Compute the dice score for each sample in the dataset and then average it"""
    y_pred_df = y_pred_df.T
    y_true_df = y_true_df.T
    individual_dice = []
    for row_index in range(y_true_df.values.shape[0]):
        dices = dice_multiclass(y_true_df.values[row_index].ravel(), y_pred_df.values[row_index].ravel())
        individual_dice.append(dices)
        # if row_index % 100 == 0:
        #     print("Processes", row_index, "pixels")


    final = np.stack(individual_dice)
    print(final.shape)
    # First, average over images for each class
    # Then, average over classes
    cls_dices = np.nanmean(final, axis=0)
    print(cls_dices.shape)
    return float(np.nanmean(cls_dices))


def criterion(preds, masks):
    # criterion_dice = DiceLoss()
    criterion_dice = DiceLossWithLogtis()
    # # criterion_ce  = BCELoss()
    # criterion_ce  = CrossEntropyLoss()

    c_dice = criterion_dice(preds, masks)
    # c_ce = criterion_ce(preds, masks)
    # return 0.5*c_dice + 0.5*c_ce
    return c_dice