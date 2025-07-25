import os
import sys
from globalss import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import SOCDataset
from UNet_Model import UNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_MODEL_PATH = OUTPUT_DIR / "unet_model.pt"

def main():
    log_file = open(OUTPUT_DIR / 'UNet_log.txt', 'a')
    
    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=log_file)
        
    def check_soc_dataset(dynamic_path, static_path):
        dynamic_data = np.load(dynamic_path)
        static_data = np.load(static_path)
        
        log_print("Dynamic data keys:", list(dynamic_data.keys()))
        log_print("Static data keys:", list(static_data.keys()))
        
        for key in ['soc_fast', 'soc_slow', 'v_fast', 'v_slow', 'precip', 'check_dams']:
            log_print(f"{key} shape:", dynamic_data[key].shape, f", avg = {np.nanmean(dynamic_data[key][0])}, max = {np.nanmax(dynamic_data[key][0])}, min = {np.nanmin(dynamic_data[key][0])}")
        
        for key in ['dem', 'loess_border_mask', 'river_mask', 'small_boundary_mask',
                    'large_boundary_mask', 'small_outlet_mask', 'large_outlet_mask']:
            log_print(f"{key} shape:", static_data[key].shape)
    # check_soc_dataset(PROCESSED_DIR / "dynamic_data.npz", PROCESSED_DIR / "static_data.npz")
        
    if DEVICE.type == 'cuda':
        log_print("Using CUDA:", torch.cuda.get_device_name(DEVICE))
    else:
        log_print("Using CPU")


    dataset = SOCDataset(
        dynamic_path=PROCESSED_DIR / "dynamic_data.npz",
        static_path=PROCESSED_DIR / "static_data.npz"
    )
    dataloader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,)

    log_print(f"Data loading completed")
    # check_soc_dataset(PROCESSED_DIR / "dynamic_data.npz", PROCESSED_DIR / "static_data.npz")

    model = UNet(in_channels=13, out_channels=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ------------------------ Loss Function ------------------------
    def masked_mse(pred, target, river_mask):
        """
        Compute MSE only for non-river pixels
        pred, target: [B, 2, H, W]
        river_mask: [1, H, W] — broadcast to batch
        """
        valid_mask = (river_mask == 0).to(pred.device)  # output: 1 = valid pixel
        valid_mask = valid_mask.unsqueeze(0).expand(pred.size(0), 1, -1, -1)  # [B, 1, H, W]
        valid_mask = valid_mask.expand(-1, 2, -1, -1)  # [B, 2, H, W]

        loss = ((pred - target) ** 2) * valid_mask
        return loss.sum() / valid_mask.sum().clamp(min=1)

    # ------------------------ Training Loop ------------------------
    start_epoch = 0
    if os.path.exists(SAVE_MODEL_PATH):
        ckpt = torch.load(SAVE_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        loss_history = ckpt.get('loss_history', [])
        log_print(f"Resuming from epoch {start_epoch}")

    else:
        loss_history = []
        
    total_start = time.time()
    log_print(f"Start Training...")
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        total_loss = 0.0
        
        for step, (x, y ) in enumerate(dataloader):
            step_start = time.time()
            if torch.isnan(x).any() or torch.isinf(x).any():
                log_print("NaN or Inf detected in input x")
            if torch.isnan(y).any() or torch.isinf(y).any():
                log_print("NaN or Inf detected in target y")

            x = x.to(DEVICE)    # [B, 8, H, W]
            y = y.to(DEVICE)    # [B, 2, H, W]
            
            # Extract river_mask from input channels
            river_mask = x[:, -1]   # [B, H, W]
            river_mask = river_mask[0] # Assume all river masks are the same (static)
            
            # Forward pass
            pred = model(x)
            loss = masked_mse(pred, y, river_mask)
            
            if torch.isnan(loss) or torch.isinf(loss):
                log_print("NaN or Inf detected in loss at Step", step+1)
                break
                
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    log_print(f"NaN or Inf detected in gradient of {name}")
            
            total_loss += loss.item()
            
            if (step + 1) % PRINT_FREQ == 0:
                log_print(f"Epoch {epoch+1} | Step {step+1}/{len(dataloader)} | Loss: {loss.item():.6f} | Time: {time.time() - step_start:.2f}s")
                
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': loss_history
        }, SAVE_MODEL_PATH)
        log_print(f"Checkpoint saved at epoch {epoch+1}")
        
        # ------------------------ Plot Loss ------------------------
        plt.plot(range(1, epoch + 2), loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Training Loss Curve')
        plt.grid()
        plt.savefig(OUTPUT_DIR / "UNet_Loss_Curve.png")
        plt.close()

        # ------------------------ Save Model -----------------------
        torch.save(model.state_dict(), SAVE_MODEL_PATH)
        log_print(f"Model saved to {SAVE_MODEL_PATH}")
        
        log_print(f"Epoch [{epoch+1}/{NUM_EPOCHS}, Avg Loss: {avg_loss:.6f}] | Completed in {time.time() - epoch_start:.2f}s")

        log_print(f"Total Training Time: {time.time() - total_start:.2f} sec")

        log_file.close()
    
if __name__ == "__main__":
    # Windows 下必须调用
    from multiprocessing import freeze_support
    freeze_support()
    main()


