import os
import sys
from globalss import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import SOCDataset
from UNet_Model import UNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_MODEL_PATH = OUTPUT_DIR / "unet_model.pt"

dataset = SOCDataset(
    dynamic_path=PROCESSED_DIR / "dynamic_data.npz",
    static_path=PROCESSED_DIR / "static_data.npz"
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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
model.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    
    for step, (x, y ) in enumerate(dataloader):
        x = x.to(DEVICE)    # [B, 8, H, W]
        y = y.to(DEVICE)    # [B, 2, H, W]
        
        # Extract river_mask from input channels
        river_mask = x[:, -1]   # [B, H, W]
        river_mask = river_mask[0] # Assume all river masks are the same (static)
        
        # Forward pass
        pred = model(x)
        loss = masked_mse(pred, y, river_mask)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (step + 1) % PRINT_FREQ == 0:
            print(f"Epoch {epoch+1} | Step {step+1}/{len(dataloader)} | Loss: {loss.item():.6f}")
            
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}, Avg Loss: {avg_loss:.6f}]")

# ------------------------ Save Model -----------------------
torch.save(model.state_dict(), SAVE_MODEL_PATH)
print(f"Model saved to {SAVE_MODEL_PATH}")
