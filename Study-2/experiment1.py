# this experiment is a part of study-1, trying the use of anisotropic pooling 
# TODO 1.  TRAIN A RANDOM TUNER to and try to find the mini
import os
import numpy as np
import zarr
import torch
from pathlib import Path
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits as BCE
from torch.utils.data import Dataset
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
from torch.utils.data import DataLoader
from torch import optim
import json



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# & data loader functions
def read_array_meta(dataset_zarr_path: Path):

    dataset_zarr_path = Path(dataset_zarr_path) / "0" / "zarr.json"

    with open(dataset_zarr_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta

def estimated_cells(dataset_geff_path: Path):
    dataset_geff_path = Path(dataset_geff_path)/ "0" / "zarr.json"

    with open(dataset_geff_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta["extra"]["estimated_number_of_nodes"]

def read_geff_nodes(geff_path):

    geff_path = Path(geff_path)
    label_group = zarr.open_group(str(geff_path), mode="r")
    
    node_ids = label_group["nodes/ids"][:]
    nodes_t = label_group["nodes/props/t/values"][:]
    nodes_z = label_group["nodes/props/z/values"][:]
    nodes_y = label_group["nodes/props/y/values"][:]
    nodes_x = label_group["nodes/props/x/values"][:]
    
    return {
        "ids": node_ids,
        "t": nodes_t,
        "z": nodes_z,
        "y": nodes_y,
        "x": nodes_x
    }

def load_volume(zarr_path, t_idx=None):

    zarr_path = Path(zarr_path)
    img_group = zarr.open_group(str(zarr_path), mode="r")
    img_array = img_group["0"]  # Primary image array path

    vol = img_array[t_idx]

    return vol
def normalize(vol):
    """
    Normalizes a 3D microscopy volume using 2nd and 98th percentiles 
    to handle outliers, then scales the values to the range [0, 1].
    """
    vol = vol.astype(np.float32)
    
    # Calculate 2nd and 98th percentiles to avoid extreme intensity outliers
    p_low, p_high = np.percentile(vol, (2, 98))
    
    # Clip values outside this range
    vol = np.clip(vol, p_low, p_high)
    
    # Scale safely to [0, 1] (preventing division by zero if the slice is completely blank)
    if p_high - p_low > 1e-5:
        vol = (vol - p_low) / (p_high - p_low)
    else:
        vol = np.zeros_like(vol)
        
    return vol
def heatmap_generator(shape, pts, cell_radius_um=5, dz=1.625, dy=0.40625, dx=0.40625):
    """
    Generates a 3D target heatmap containing multiple cell centers for a single frame.
    
    Args:
        shape (tuple): (Z, Y, X) dimensions of the volume patch or full frame.
        pts (np.ndarray): Array of shape (N, 3) containing [z, y, x] coordinates of all cells in frame `t`.
        cell_radius_um (float): Estimated physical radius of the cells.
        dz, dy, dx (float): Physical voxel spacing in micrometers.
        
    Returns:
        np.ndarray: 3D float32 array of shape (Z, Y, X) with values ranging from 0.0 to 1.0.
    """
    # 1. Initialize an empty canvas for the frame
    canvas = np.zeros(shape, dtype=np.float32)
    
    if len(pts) == 0:
        return canvas  # Return blank canvas if no cells are present in this frame
        
    # 2. Create global coordinate grid once for efficiency
    Z, Y, X = shape
    zz, yy, xx = np.ogrid[:Z, :Y, :X]
    
    sigma_physical = cell_radius_um / 2.0
    """debug_deno = []
    debug_gauss = []
    debug_num = []"""
    
    # 3. Loop through every cell node in this frame and paint its Gaussian
    for pt in pts:
        cz, cy, cx = pt
        
        # Using Gaussian formula for heatmap generation
        dist_sq = (((zz - cz) * dz) ** 2) / (2 * sigma_physical ** 2) + \
                  (((yy - cy) * dy) ** 2) / (2 * sigma_physical ** 2) + \
                  (((xx - cx) * dx) ** 2) / (2 * sigma_physical ** 2)
        
        # Generate the Gaussian blob for this cell
        gaussian = np.exp(-dist_sq)
        
        # Merge onto the canvas using max (prevents value clipping/overflow on overlap)
        canvas = np.maximum(canvas, gaussian)
        
    return canvas  
def weight_map_gen(target_vol, input_vol, bg_wt = 0.0 ,low_wt = 0.1 ,bg_thesh = 0.30 ,cell_thresh = 0.1 ,cell_wt = 5):
    
    # 1. Initialize entire patch with the neutral/ignore weight
    weight_map = np.full(target_vol.shape, bg_wt, dtype=np.float32)
    
    # 2. Confident background: low optical intensity
    bg_cutoff = float(np.quantile(input_vol, bg_thesh))
    weight_map[input_vol < bg_cutoff] = low_wt
    
    # 3. Confident positives: annotated Gaussian centers override background
    weight_map[target_vol > cell_thresh] = cell_wt
    
    return weight_map



class ConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding='same', stride=1)
        self.batch1 = nn.BatchNorm3d(mid_channels)
        self.activation1 = nn.ReLU() 
        
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding='same', stride=1)
        self.batch2 = nn.BatchNorm3d(out_channels)
        self.activation2 = nn.ReLU()
        
    def forward(self, X):
        out1 = self.activation1(self.batch1(self.conv1(X)))
        out2 = self.activation2(self.batch2(self.conv2(out1)))
        return out2

class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Hybrid code to increse z and decrese the x and y axis axis to `128`
        self.stem = nn.Sequential(
            nn.ConvTranspose3d(1, 32, kernel_size=(2,1,1), stride=(2,1,1), padding=0),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 32, kernel_size=(1,2,2), stride=(1,2,2)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        
        # 2. Encoder Path (operates safely at half resolution)
        self.enc1 = ConvBlock(32, 32, 64)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock(64, 64, 128) # Bottleneck
        
        # 3. Decoder Path
        self.TransConv1 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.decod1 = ConvBlock(192, 64, 64) # 128 (from decoder) + 64 (from skip l1) = 192
        
        # 4. Final Upsample (Restores output back to full original size)
        self.final_upsample = nn.ConvTranspose3d(64, 16,kernel_size=(1, 2, 2),stride=(1, 2, 2))
        self.depth_reduce = nn.Conv3d(16, 16,kernel_size=(2, 1, 1),stride=(2, 1, 1))
        self.finalconv = nn.Conv3d(16, 1,kernel_size=1, padding='same', stride=1)
        
    def forward(self, X):
        print("Input:             ", tuple(X.shape))
    
        stem_out = self.stem(X)
        print("Stem:              ", tuple(stem_out.shape))
    
        l1 = self.enc1(stem_out)
        print("Encoder 1 (enc1):   ", tuple(l1.shape))
    
        m1 = self.pool1(l1)
        print("MaxPool (pool1):    ", tuple(m1.shape))
    
        l2 = self.enc2(m1)
        print("Encoder 2 (enc2):   ", tuple(l2.shape))
    
        t1 = self.TransConv1(l2)
        print("TransConv1:         ", tuple(t1.shape))
    
        c1 = torch.cat((l1, t1), dim=1)
        print("Skip concatenation: ", tuple(c1.shape))
    
        d1 = self.decod1(c1)
        print("Decoder (decod1):   ", tuple(d1.shape))
    
        restored = self.final_upsample(d1)
        print("In-plane upsample:  ", tuple(restored.shape))
    
        restored = self.depth_reduce(restored)
        print("Depth reduction:   ", tuple(restored.shape))
    
        out = self.finalconv(restored)
        print("Final output:       ", tuple(out.shape))
    
        return out

def weighted_loss(pred_vol, target_vol, weight_map):
    loss = BCE(pred_vol, target_vol, reduction="none")
    weighted = loss * weight_map
    return weighted.sum() / torch.clamp(weight_map.sum(), min=1.0) # to reduice the impace of the weight multiplication which increses the loss value

class CellDataset(Dataset):

    def __init__(
        self,
        DATA_DIR,
        movies,
        fpm=20
    ):
        self.DATA_DIR = Path(DATA_DIR)
        self.fpm = fpm
        self.samples = []
        self.movies = movies

        self._build_index()
        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def _build_index(self):

        if self.movies is None:
            tqdm.write("Got no files ----------------------------------")
            zarr_files = sorted(self.DATA_DIR.glob("*.zarr"))
        else:
            zarr_files = sorted(self.movies)

        for z_path in zarr_files:

            # Get movie name from the zarr filename
            ds_name = z_path.stem

            # Corresponding GEFF file
            g_path = self.DATA_DIR / f"{ds_name}.geff"

            # Require matching GEFF file
            if not g_path.exists():
                continue

            try:
                # Only read metadata, not the volume itself
                meta = read_array_meta(str(z_path))

                # Read node table
                nodes = read_geff_nodes(str(g_path))

                # Frames actually containing GEFF annotations
                node_t = nodes["t"]

                available_times = sorted(
                    set(node_t.tolist()) &
                    set(range(meta["shape"][0]))
                )

                if not available_times:
                    continue

                # Select fpm frames uniformly
                indices = np.linspace(
                    0,
                    len(available_times) - 1,
                    min(self.fpm, len(available_times))
                ).round().astype(int)

                for idx in indices:
                    t = available_times[idx]
                    self.samples.append((ds_name, t))

            except Exception as e:
                tqdm.write(
                    f"Skipping {ds_name} due to read error: {e}"
                )

        tqdm.write(
            f"[Dataset] Indexed {len(self.samples)} "
            f"paired frames across {len(zarr_files)} movies."
        )

    def __getitem__(self, idx):

        ds_name, t = self.samples[idx]

        z_path = str(self.DATA_DIR / f"{ds_name}.zarr")
        g_path = str(self.DATA_DIR / f"{ds_name}.geff")

        vol = load_volume(z_path, t)
        x = normalize(vol).astype(np.float32)

        nodes = read_geff_nodes(g_path)

        mask = nodes["t"] == t

        pts = np.stack(
            [
                nodes["z"][mask],
                nodes["y"][mask],
                nodes["x"][mask]
            ],
            axis=-1
        ).astype(np.float32)

        target_heatmap = heatmap_generator(
            x.shape,
            pts
        )

        weight_map = weight_map_gen(
            target_heatmap,
            x
        )

        return (
            torch.from_numpy(x).unsqueeze(0),
            torch.from_numpy(target_heatmap).unsqueeze(0),
            torch.from_numpy(weight_map).unsqueeze(0)
        )

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    Collects individual batch losses, computes their mean, and returns it.
    """
    model.train()
    batch_losses = []

    pbar = tqdm(dataloader, desc="Training Batch", leave=False, dynamic_ncols=True)
    
    # FIX 1: Unpack 3 items (input, target, weight_map)
    for X_batch, y_batch, w_batch in pbar:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        w_batch = w_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        
        # FIX 2: Pass all 3 arguments to the weighted_loss criterion
        loss = criterion(y_pred, y_batch, w_batch)
        loss.backward()
        optimizer.step()

        # Detach and convert to Python float to avoid retaining computation graph
        batch_loss = loss.item()
        batch_losses.append(batch_loss)
        pbar.set_postfix(batch_loss=f"{batch_loss:.4f}")

    # Compute the mean of all individual batch losses
    mean_batch_loss = float(np.mean(batch_losses))
    return mean_batch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model across batches in validation mode and returns the mean loss.
    """
    model.eval()
    val_losses = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation Batch", leave=False, dynamic_ncols=True)
        # FIX 1: Unpack 3 items (input, target, weight_map)
        for X_batch, y_batch, w_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            w_batch = w_batch.to(device)
            
            y_pred = model(X_batch)
            
            # FIX 2: Pass all 3 arguments to the weighted_loss criterion
            loss = criterion(y_pred, y_batch, w_batch)
            val_losses.append(loss.item())

    return float(np.mean(val_losses))


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    max_epochs=20,
    patience=4,
    min_delta=1e-4,
    device="cpu" # ^ fallback
):
    """
    Trains with early stopping and returns the best model along with
    the mean train and validation loss for each epoch.
    """
    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    train_loss_history = []
    val_loss_history = []

    epoch_pbar = tqdm(range(1, max_epochs + 1), desc="Epochs", leave=False, dynamic_ncols=True)
    for epoch in epoch_pbar:
        # Calculate mean loss of the batches for this epoch
        train_mean_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_mean_loss = evaluate(model, val_loader, criterion, device)

        train_loss_history.append(train_mean_loss)
        val_loss_history.append(val_mean_loss)

        # Early stopping check using min_delta tolerance
        if (best_val_loss - val_mean_loss) > min_delta:
            best_val_loss = val_mean_loss            
            best_weights = copy.deepcopy(model.state_dict())
            
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_pbar.set_postfix(
            train_loss=f"{train_mean_loss:.4f}",
            val_loss=f"{val_mean_loss:.4f}",
            best_val=f"{best_val_loss:.4f}",
            patience=f"{patience_counter}/{patience}"
        )

        if patience_counter >= patience:
            tqdm.write(f"\n[Early Stopping] Triggered at epoch {epoch}, Lowest Val Loss: {best_val_loss:.4f}")
            break

    # Restore optimal model state
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return (
        model,
        train_loss_history,
        val_loss_history
    )

# ! for kaggle  = /kaggle/input/competitions/biohub-cell-tracking-during-development/train
# ! for kaggle = /kaggle/working
def main():
    
    DATA_DIR = Path(" /kaggle/input/competitions/biohub-cell-tracking-during-development/train") # 
    OUTPUT_DIR = Path("Study-2/results")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR = OUTPUT_DIR / "weights"
    METRICS_DIR = OUTPUT_DIR / "metrics"

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    zarr_files = sorted(list(DATA_DIR.glob("*.zarr")))
    split_idx = int(len(zarr_files) * 0.8)
    train_movies = zarr_files[:split_idx]
    val_movies = zarr_files[split_idx:]

    train_sizes = [20]
    train_errors = []
    val_errors = []
    experiment_history = {}

    val_dataset = CellDataset(
                DATA_DIR=DATA_DIR,
                movies=val_movies,
                fpm=10 # ! can be adjusted
            )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory= True)
    
    for n_movies in train_sizes:

        selected_train_movies = train_movies[:n_movies]

        train_dataset = CellDataset(
            DATA_DIR=DATA_DIR,
            movies=selected_train_movies,
            fpm= 20 # ! can be adjusted
        )

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory= True)

        model = UNet3D().to(device)

        optimizer = optim.NAdam(
            model.parameters(),
            lr=1e-4,            # Baseline learning rate
            weight_decay=0.0,   # Zero weight decay as requested
            betas=(0.9, 0.999)  # Standard default momentum parameters
        )

        model, train_loss_history, val_loss_history = train_with_early_stopping(
            model, train_loader, val_loader, weighted_loss, optimizer,
            max_epochs=25, patience=5, min_delta=1e-4, device=device
        )

        mean_train_loss = float(np.mean(train_loss_history))
        mean_val_loss = float(np.mean(val_loss_history))
        train_errors.append(mean_train_loss)
        val_errors.append(mean_val_loss)

        weight_file = WEIGHTS_DIR / f"unet3d_v2_{n_movies}.pt"
        torch.save(model.state_dict(), weight_file)

        # 2. Store detailed history for JSON export
        experiment_history[n_movies] = {
            "train_loss_per_epoch": train_loss_history,
            "val_loss_per_epoch": val_loss_history,
            "epochs_completed": len(train_loss_history),
            "weights_file": weight_file.name
        }

        del train_loader
        del train_dataset
        del model
        del optimizer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Save complete experiment history
    result_file = METRICS_DIR / "metric.json"

    with open(result_file, "w") as f:
        json.dump(experiment_history, f, indent=4)

if __name__ == "__main__":
    main()
