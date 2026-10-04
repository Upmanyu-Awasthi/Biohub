import torch
import zarr
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import Subset
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
# & Dataset defination
import json
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

class CellDataset(Dataset):

    def __init__(
        self,
        data_dir,
        fpm=20,
        augment=False
    ):
        self.data_dir = Path(data_dir)
        self.fpm = fpm
        self.augment = augment
        self.samples = []
        self._build_index()
        self.samples.sort()

    def __len__(self):
        return len(self.samples)
    
    def _build_index(self):
        zarr_files = sorted(self.data_dir.glob("*.zarr"))

        for z_path in zarr_files:
            ds_name = z_path.stem
            g_path = self.data_dir / f"{ds_name}.geff"

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
                )

                for idx in indices:
                    t = available_times[idx]
                    self.samples.append((ds_name, t))

            except Exception as e:
                print(
                    f"Skipping {ds_name} due to read error: {e}"
                )

        # ! i had added sample sorting i guess it is repeated

        print(
            f"[Dataset] Indexed {len(self.samples)} "
            f"paired frames across all movies."
        )

    def __getitem__(self, idx):

        ds_name, t = self.samples[idx]

        z_path = str(self.data_dir / f"{ds_name}.zarr")
        g_path = str(self.data_dir / f"{ds_name}.geff")

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
            torch.from_numpy(weight_map).unsqueeze(0),
            torch.from_numpy(pts)
        )
from torch.utils.data import Subset
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from torch import optim
dataset = CellDataset(
    data_dir= "/home/upmanyu/ml-workspace/Biohub/data/biohub-cell-tracking-during-development/train",
    fpm= 5, # ! adjust for lower traning time
    augment= False
)
groups = np.array([movie for movie, frame in dataset.samples])
gkf = GroupKFold(n_splits=5, random_state = 47, shuffle = True)
for fold, (train_idx, val_idx) in enumerate(gkf.split(dataset, groups=groups)):
    print(f"--- Training on Fold {fold+1}/5 ---")
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    

# & ----------------------Model architecture--------------------------------------------
import torch
import torch.nn as nn

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
        
        # 1. Entry Stem (Your idea: learned downsampling to save VRAM)
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1), # Halves spatial & depth size
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
        self.final_upsample = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.finalconv = nn.Conv3d(64, 1, kernel_size=1, padding='same', stride=1)
        
    def forward(self, X):
        # Input: (B, 1, 64, 256, 256)
        
        # Step 1: Entry Stem Compression
        stem_out = self.stem(X)          # Shape: (B, 32, 32, 128, 128)
        
        # Step 2: Encoder
        l1 = self.enc1(stem_out)         # Shape: (B, 64, 32, 128, 128) -> Saved for skip connection
        m1 = self.pool1(l1)              # Shape: (B, 64, 16, 64, 64)
        l2 = self.enc2(m1)               # Shape: (B, 128, 16, 64, 64) -> Bottleneck feature map
        
        # Step 3: Decoder & Skip Connection Integration
        t1 = self.TransConv1(l2)         # Shape: (B, 128, 32, 128, 128)
        c1 = torch.cat((l1, t1), dim=1)  # Shape: (B, 192, 32, 128, 128) -> Combined features!
        d1 = self.decod1(c1)             # Shape: (B, 64, 32, 128, 128)
        
        # Step 4: Restore Original Resolution
        restored = self.final_upsample(d1) # Shape: (B, 64, 64, 256, 256)
        out = self.finalconv(restored)     # Shape: (B, 1, 64, 256, 256)
        
        return out
# &---------------------------------------- end of model architecture ----------------------------------------- 

def objective(trial):
    # 1. Suggest a static learning rate on a log scale
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D().to(device)  
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.NAdam(
        model.parameters(),
        lr=lr,
        weight_decay=0.0,
        betas=(0.9, 0.999)
    )
    
    epochs = 8  # Optimized proxy epoch count
    
    # 2. Outer loop for epochs with TQDM description
    epoch_bar = tqdm(range(epochs), desc=f"Trial {trial.number} (lr={lr:.5f})", leave=False)
    
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0
        
        # Inner loop for training batches with live TQDM progress
        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for data, targets, weigths in batch_bar:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets, weigths)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_data, val_targets, weigths in val_loader:
                val_data, val_targets = val_data.to(device), val_targets.to(device)
                val_outputs = model(val_data)
                v_loss = criterion(val_outputs, val_targets, weigths)
                val_loss += v_loss.item()
                
        mean_val_loss = val_loss / len(val_loader)
        epoch_bar.set_postfix(val_loss=f"{mean_val_loss:.4f}")
        
        # Report to Optuna for trial pruning
        trial.report(mean_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return mean_val_loss

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=10)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=10)

# 4. Run Bayesian Optimization Study
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=15, timeout=7200)

print("\nBest Hyperparameter Found:")
for key, value in study.best_params.items():
    print(f"  * {key}: {value}")
print(f"Best Validation Loss: {study.best_value:.4f}")