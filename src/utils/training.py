import torch
import numpy as np
import tqdm
import os
import json
from torch.utils.tensorboard import SummaryWriter
import losses

def validate(model, criterion, vl_dataloader, device, writer=None, epoch=0):
    model.eval()
    with torch.no_grad():
                
        losses_val = []
        cnt = 0.
        dices = []
        total_batches = []
        total_preds = []
        for batch, batch_data in enumerate(vl_dataloader):

            imgs, msks = batch_data
            
            
            cnt += msks.shape[0]
            
            imgs = imgs.to(device)
            msks = msks.to(device)
            total_batches.append(msks)
            
            preds = model(imgs)
            loss = criterion(preds, msks)
            losses_val.append(loss.item())
            

            torch.mps.empty_cache()

            preds_argmax = torch.argmax(preds, dim=1)
            total_preds.append(preds_argmax)
            if batch == 0:
                #normalize the image between 0 and 1
                img_tensorboard = (imgs[0] - imgs[0].min())/(imgs[0].max() - imgs[0].min())
                msk_tensorboard = (msks[0] - msks[0].min())/(msks[0].max() - msks[0].min())
                pred_tensorboard = (preds_argmax[0] - preds_argmax[0].min())/(preds_argmax[0].max() - preds_argmax[0].min())
                writer.add_image(f"Val/Images/", img_tensorboard, epoch)
                writer.add_image(f"Val/Masks/", msk_tensorboard.unsqueeze(0), epoch)
                writer.add_image(f"Val/Preds/", pred_tensorboard.unsqueeze(0), epoch)
        
        total_preds = torch.cat(total_preds)
        total_batches = torch.cat(total_batches)
        dice_score = losses.dice_pandas(total_batches, total_preds)
        

        
#             _cml = f"curr_mean-loss:{np.sum(losses)/cnt:0.5f}"
#             _bl = f"batch-loss:{losses[-1]/msks.shape[0]:0.5f}"
#             iterator.set_description(f"Validation) batch:{batch+1:04d} -> {_cml}, {_bl}")
        
        # print the final results
        loss = np.sum(losses_val)/cnt
    
    return loss, dice_score

# import gc

def train(
    model, 
    device, 
    tr_dataloader,
    vl_dataloader,

    epochs,
    
    criterion,
    optimizer,
    scheduler,
    
    save_dir='./',
    save_file_id=None,
    writer=None
):

    
    torch.cuda.empty_cache()
    model = model.to(device)
    
    epochs_info = []
    best_model = None
    best_result = {}
    best_vl_loss = np.inf
    for epoch in range(epochs):
        model.train()
        
        tr_iterator = tqdm.tqdm(enumerate(tr_dataloader))
        tr_losses = []
        cnt = 0
        for batch, batch_data in tr_iterator:
            imgs, msks = batch_data
            
            imgs = imgs.to(device)
            msks = msks.to(device)

            # optimizer.zero_grad()
            
            preds = model(imgs)
            # print(preds.shape, msks.shape)
            loss = criterion(preds, msks)
            # gc.collect()
            torch.mps.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # if batch % 4 == 0:
            
            optimizer.step()
            
            
            cnt += imgs.shape[0]
            tr_losses.append(loss.item())
            
            # write details for each training batch
            _cml = f"curr_mean-loss:{np.sum(tr_losses)/cnt:0.5f}"
            _bl = f"mean_batch-loss:{tr_losses[-1]/imgs.shape[0]:0.5f}"
            tr_iterator.set_description(f"Training) ep:{epoch:03d}, batch:{batch+1:04d} -> {_cml}, {_bl}")
            if writer:
                writer.add_scalar("Loss/train", loss.item(), epoch*len(tr_dataloader)+batch)
                preds_argmax = torch.argmax(preds, dim=1)
                if batch == 0:
                    #normalize the image between 0 and 1
                    img_tensorboard = (imgs[0] - imgs[0].min())/(imgs[0].max() - imgs[0].min())
                    msk_tensorboard = (msks[0] - msks[0].min())/(msks[0].max() - msks[0].min())
                    pred_tensorboard = (preds_argmax[0] - preds_argmax[0].min())/(preds_argmax[0].max() - preds_argmax[0].min())
                    writer.add_image(f"Train/Images/", img_tensorboard, epoch)
                    writer.add_image(f"Train/Masks/", msk_tensorboard.unsqueeze(0), epoch)
                    writer.add_image(f"Train/Preds/", pred_tensorboard.unsqueeze(0), epoch)
                if batch % 50 == 0:
                    # print("batch", batch)
                    dice_score = losses.dice_pandas(msks, preds_argmax)
                    # print("dice_score", dice_score)
                    writer.add_scalar("Dice/train", dice_score, epoch*len(tr_dataloader)+batch)

                
                    

        
        tr_loss = np.sum(tr_losses)/cnt
        
        # validate model
        vl_loss, vl_dice_score = validate(model, criterion, vl_dataloader, device, writer, epoch)

        if writer:
            writer.add_scalar("Loss/val", vl_loss, epoch)
            writer.add_scalar("Dice/val", vl_dice_score, epoch)

        if vl_loss < best_vl_loss:
            # find a better model
            best_model = model
            best_vl_loss = vl_loss
            best_result = {
                'tr_loss': tr_loss,
                'vl_loss': vl_loss,
            }
        
        # write the final results
        epoch_info = {
            'tr_loss': tr_loss,
            'vl_loss': vl_loss,
        }
        epochs_info.append(epoch_info)
        # save model's state_dict
        fn = "last_model_state_dict.pt"
        fp = os.path.join(save_dir,fn)
        torch.save(model.state_dict(), fp)
        
        # save the best model's state_dict
        fn = "best_model_state_dict.pt"
        fp = os.path.join(save_dir, fn)
        torch.save(best_model.state_dict(), fp)
#         epoch_tqdm.set_description(f"Epoch:{epoch+1}/{EPOCHS} -> tr_loss:{tr_loss}, vl_loss:{vl_loss}")
    
        scheduler.step(vl_loss)
  
    # save final results
    res = {
        'id': save_file_id,
        'epochs_info': epochs_info,
        'best_result': best_result
    }
    fn = f"{save_file_id+'_' if save_file_id else ''}result.json"
    fp = os.path.join(save_dir,fn)
    with open(fp, "w") as write_file:
        json.dump(res, write_file, indent=4)


    
    return best_model, model, res