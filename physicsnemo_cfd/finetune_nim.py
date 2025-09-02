# Standard library imports
import os
import yaml
from pathlib import Path
from types import SimpleNamespace

# Third-party imports
import numpy as np
import torch
from tqdm import tqdm

# Torch utilities
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel

# Project-specific imports (modulus)
from modulus.launch.utils import save_checkpoint
from modulus.datapipes.cae.domino_datapipe import DoMINODataPipe
from modulus.models.domino.model import DoMINO
from modulus.utils.domino.utils import *

def setup_distributed():
    """
    Initialize distributed training environment.
    
    Returns:
        tuple: (device, rank, world_size)
            - device: torch.device for computation
            - rank: process rank in distributed setup
            - world_size: total number of processes
    """
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # Set up CUDA device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Initialize distributed process group if needed
    if world_size > 1:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    return device, rank, world_size

def mse_loss_fn(output, target, padded_value=-10):
    """
    Compute masked MSE loss, ignoring padded values.
    
    Args:
        output (torch.Tensor): Model predictions
        target (torch.Tensor): Ground truth values
        padded_value (float): Value used for padding (default: -10)
    
    Returns:
        torch.Tensor: Mean squared error loss
    """
    # Move target to same device as output
    target = target.to(output.device)
    # Create mask for non-padded values
    mask = torch.abs(target - padded_value) > 1e-3
    # Compute masked loss
    masked_loss = torch.sum(((output - target) ** 2) * mask) / torch.sum(mask)
    return masked_loss.mean()

def create_dataset(phase):
    """
    Create DoMINO dataset for specified phase (train/val).
    
    Args:
        phase (str): Dataset phase ('train' or 'val')
    
    Returns:
        DoMINODataPipe: Configured dataset
    """
    return DoMINODataPipe(
        DATA_PATHS[phase], 
        phase=phase, 
        grid_resolution=model_cfg["interp_res"],
        surface_variables=SURFACE_VARS, 
         model_type=MODEL_TYPE, 
        normalize_coordinates=True,
        sampling=True, 
        sample_in_bbox=True, 
        volume_points_sample=model_cfg,
        surface_points_sample=model_cfg["surface_points_sample"], 
        geom_points_sample=model_cfg["geom_points_sample"],
        positional_encoding=False, 
        scaling_type=model_cfg["normalization"],
        bounding_box_dims=BOUNDING_BOX,
        bounding_box_dims_surf=BOUNDING_BOX_SURF,
        num_surface_neighbors=model_cfg["num_surface_neighbors"], 
        surface_factors=np.load(SURF_SAVE_PATH),
        #compute_scaling_factors=True,
    )

def create_dataloaders(rank, world_size):
    """
    Create train and validation dataloaders with distributed sampling.
    
    Args:
        rank (int): Process rank
        world_size (int): Total number of processes
    
    Returns:
        tuple: (train_loader, val_loader, train_sampler, val_sampler)
    """
    # Create datasets
    train_dataset, val_dataset = create_dataset("train"), create_dataset("val")
    
    # Configure distributed samplers if needed
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    
    # Create dataloaders
    return (
        DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, shuffle=train_sampler is None),
        DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, shuffle=False),
        train_sampler, val_sampler
    )

def create_model(device, rank, world_size):
    """
    Create and configure DoMINO model with distributed training support.
    
    Args:
        device (torch.device): Computation device
        rank (int): Process rank
        world_size (int): Total number of processes
    
    Returns:
        DoMINO: Configured model (wrapped in DistributedDataParallel if world_size > 1)
    """
    # Initialize model with configuration

    model = DoMINO(
            input_features=INPUT_FEATURES,
            output_features_vol=OUTPUT_FEATURES_VOL,
            output_features_surf=NUM_SURF_VARS,
            model_parameters=SimpleNamespace(
                model_type=MODEL_TYPE,
                interp_res=model_cfg["interp_res"],
                surface_neighbors=model_cfg["use_surface_normals"],
                use_surface_normals=model_cfg["use_only_normals"],
                use_only_normals=model_cfg["use_only_normals"],
                encode_parameters=model_cfg["encode_parameters"],
                positional_encoding=model_cfg["positional_encoding"],
                integral_loss_scaling_factor=model_cfg["integral_loss_scaling_factor"],
                normalization=model_cfg["normalization"],
                use_sdf_in_basis_func=model_cfg["use_sdf_in_basis_func"],
                geometry_rep=SimpleNamespace(
                    base_filters=geo_rep["base_filters"],
                    geo_conv=SimpleNamespace(
                        base_neurons=geo_conv["base_neurons"], # we have 32, weights have 8
                        base_neurons_out=geo_conv["base_neurons_out"], 
                        radius_short=geo_conv["radius_short"],
                        radius_long=geo_conv["radius_long"],
                        hops=geo_conv["hops"],  
                    ),
                    geo_processor=SimpleNamespace(base_filters=geo_pro["base_filters"]),
                    geo_processor_sdf=SimpleNamespace(base_filters=geo_pro["base_filters"])
                ),
                nn_basis_functions=SimpleNamespace(base_layer=basis_func["base_layer"]),
                parameter_model=SimpleNamespace(
                    base_layer=param_model["base_layer"],
                    scaling_params=param_model["scaling_params"],
                ),
                position_encoder=SimpleNamespace(base_neurons=pos_enc["base_neurons"]),
                geometry_local=SimpleNamespace(
                    neighbors_in_radius=geo_local["neighbors_in_radius"],
                    radius=geo_local["radius"],
                    base_layer=geo_local["base_layer"],
                ),
                aggregation_model=SimpleNamespace(base_layer=agg_model["base_layer"])
            ),
        ).to(device)
    
    # Wrap model for distributed training if needed
    if world_size > 1:
        model = DistributedDataParallel(
            model, 
            device_ids=[rank], 
            output_device=rank, 
            find_unused_parameters=True
        )
    
    return model

def run_epoch(train_loader, val_loader, model, optimizer, scaler, device, epoch, best_vloss, rank, world_size):
    """
    Run one training epoch with validation.
    
    Args:
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        model (DoMINO): Model to train
        optimizer (torch.optim.Optimizer): Optimizer
        scaler (GradScaler): Gradient scaler for mixed precision
        device (torch.device): Computation device
        epoch (int): Current epoch number
        best_vloss (float): Best validation loss so far
        rank (int): Process rank
        world_size (int): Total number of processes
    
    Returns:
        float: Validation loss for this epoch
    """
    # Training phase
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}") if rank == 0 else train_loader
    
    for batch in pbar:
        # Move batch to device
        batch = dict_to_device(batch, device)
        
        # Forward pass with mixed precision
        with autocast():
            _, pred_surf = model(batch)
            loss = mse_loss_fn(pred_surf, batch["surface_fields"])
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Update loss tracking
        train_loss += loss.item()
        if rank == 0:
            pbar.set_postfix({
                "train_loss": f"{train_loss/(pbar.n+1):.5e}", 
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })
    
    # Compute average training loss
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        val_loss = sum(
            mse_loss_fn(model(dict_to_device(batch, device))[1], batch["surface_fields"].to(device)).item() 
            for batch in val_loader
        ) / len(val_loader)
    
    # Handle distributed training metrics
    if world_size > 1:
        avg_train_loss, val_loss = [torch.tensor(v, device=device) for v in [avg_train_loss, val_loss]]
        torch.distributed.all_reduce(avg_train_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.SUM)
        avg_train_loss, val_loss = avg_train_loss.item() / world_size, val_loss.item() / world_size
    
    # Save checkpoints on main process
    if rank == 0:
        if val_loss < best_vloss:
            save_checkpoint(
                os.path.join(MODEL_SAVE_DIR, "best_model"), 
                models=model,
                optimizer=optimizer,
                scaler=scaler
            )

        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                MODEL_SAVE_DIR, 
                models=model, 
                optimizer=optimizer, 
                scaler=scaler, 
                epoch=epoch
            )
    
    return val_loss

def train(model, device, rank, world_size):
    """
    Function that orchestrates the training process.
    Handles distributed training setup, model creation and training loop.
    """

    # Create output directory on main process
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True) if rank == 0 else None
    
    # Set up data
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(rank, world_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize learning rate scheduler and gradient scaler
    scheduler =  torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[15, 85, 150, 200, 250, 300, 350, 400], gamma=0.5
    )
    scaler = GradScaler()
    
    # Training loop
    best_vloss = float('inf')
    for epoch in range(NUM_EPOCHS):
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        best_vloss = min(
            best_vloss, 
            run_epoch(
                train_loader, val_loader, model, optimizer, 
                scaler, device, epoch, best_vloss, rank, world_size
            )
        )
        scheduler.step()

def load_model(model, device, model_path):
    """
    Load model weights from checkpoint.
    
    Args:
        model (DoMINO): Model to load weights into
        model_path (str): Path to the checkpoint file
    
    Returns:
        DoMINO: Model with loaded weights
    """
    checkpoint = torch.load(model_path,map_location=device)

    # Drop all volumne keys from the state_dict
    for key in list(checkpoint.keys()):
        if "vol" in key:
            del checkpoint[key]

    model.load_state_dict(checkpoint)
    return model

if __name__ == "__main__":

    # Project and Experiment Configuration
    EXPERIMENT_TAG = 4
    PROJECT_NAME = "ahmed_body_dataset"

    # Directory Paths
    BASE_DIR = Path("/workspace")
    OUTPUT_DIR = BASE_DIR / "outputs" / PROJECT_NAME / str(EXPERIMENT_TAG)
    DATA_DIR = Path("/workspace/data/ahmed_body")
    PROCESSED_DIR = DATA_DIR / "prepared_surface_data"
    CHECKPOINT_DIR = OUTPUT_DIR / "models"
    MODEL_SAVE_DIR = CHECKPOINT_DIR / "NIM"
    SURF_SAVE_PATH = BASE_DIR / "outputs" / PROJECT_NAME / "surface_scaling_factors.npy"

    # Dataset Paths
    DATA_PATHS = {
        "train": DATA_DIR / "train_prepared_surface_data",
        "val": DATA_DIR / "validation_prepared_surface_data",
        "test": DATA_DIR / "test_prepared_surface_data",
    }

    # Ensure directories exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Physical Variables
    SURFACE_VARS = ["p", "wallShearStress"]
    MODEL_TYPE = "surface"

    # Training Hyperparameters
    NUM_EPOCHS = 100
    LR = 1e-3
    BATCH_SIZE = 1
    GRID_RESOLUTION = [128, 64, 48]
    INPUT_FEATURES = 3
    OUTPUT_FEATURES_VOL = None
    NUM_SURFACE_NEIGHBORS = 7
    NORMALIZATION = "min_max_scaling"
    INTEGRAL_LOSS_SCALING = 0
    NUM_SURF_VARS = 4
    CHECKPOINT_INTERVAL = 1

    # Bounding Box Configuration
    BOUNDING_BOX = SimpleNamespace(
        max=[0.2, 0.5, 0.6],
        min=[-1.7, -0.3, -0.3],
    )
    BOUNDING_BOX_SURF = SimpleNamespace(
        max=[0.1, 0.4, 0.5],
        min=[-1.6, -0.2, -0.2],
    )

    # Distributed Training Configuration
    os.environ["MASTER_PORT"] = "29501"

    # cuDNN Optimisation
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Model Configuration
    CONFIG_PATH = Path("../opt/triton/model/1/conf/config.yaml")
    #MODEL_PATH = Path("../opt/nim/.cache/ngc/hub/models--nim--nvidia--domino-drivsim/blobs/9e683e43a5fbe30a84dff0a08925116d")
    MODEL_PATH = Path("outputs/ahmed_body_dataset/4/models/NIM/DoMINO.0.35.pt")

    # Load Model Configuration
    config = yaml.safe_load(CONFIG_PATH.read_text())
    model_cfg = config["model"]

    # Model Components
    geo_rep = model_cfg["geometry_rep"]
    geo_conv = geo_rep["geo_conv"]
    geo_pro = geo_rep["geo_processor"]
    geo_local = model_cfg["geometry_local"]
    param_model = model_cfg["parameter_model"]
    basis_func = model_cfg["nn_basis_functions"]
    pos_enc = model_cfg["position_encoder"]
    agg_model = model_cfg["aggregation_model"]

    print(os.getcwd())

    # Initialize distributed training
    device, rank, world_size = setup_distributed()

    # Set up model
    model = create_model(device, rank, world_size)

    # Load model weights from NIM checkpoint
    model = load_model(model, device, MODEL_PATH)

    # Run training
    train(model, device, rank, world_size)