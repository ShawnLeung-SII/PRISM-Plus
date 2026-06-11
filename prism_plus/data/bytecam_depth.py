"""ByteCameraDepth dataset loader.

Loads (RGB, sim_depth, real_depth, hole_mask, value_weight) tuples from the
ByteCameraDepth dataset (RealSense D435 paired with simulation).

Used by both PRISM (ICML 2026) and PRISM+ (TPAMI extension).

Expected on-disk layout (data_root)::

    rgb/                     # PNG, 3 channel
    sim_depth/               # 16-bit PNG or NPY, meters (after /depth_scale)
    real_depth/              # same format as sim_depth
    processed/
        hole_mask/           # pre-computed binary mask (1 = hole)
        value_weight/        # pre-computed noise weight in [0, 1]
    splits/
        train.txt            # one sample id per line
        val.txt
        test.txt
"""
import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from typing import Dict, Optional, Tuple, List
import random


# ============================================================================
# Constants
# ============================================================================
MIN_VALID_DEPTH = 0.001  # 1mm
MAX_VALID_DEPTH = 10.0   # 10m


# ============================================================================
# Utility Functions
# ============================================================================

def log_scale_depth(depth, d_min=0.05, d_max=5.0, filter_max=4.9):
    """
    Log-scale depth normalization (from StableS2R)
    """
    valid_mask = (depth >= d_min) & (depth <= filter_max)
    depth_log = np.zeros_like(depth)
    depth_log[valid_mask] = np.log(depth[valid_mask] / d_min) / np.log(d_max / d_min)
    depth_log[~valid_mask] = 1.0
    return depth_log


def compute_hole_mask_online(depth_m, min_depth=MIN_VALID_DEPTH, max_depth=MAX_VALID_DEPTH):
    """
    Compute hole mask on-the-fly (if not pre-computed)
    
    Args:
        depth_m: Depth in meters
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
    
    Returns:
        hole_mask: Binary mask (1=hole, 0=valid)
    """
    hole_mask = ((depth_m < min_depth) | (depth_m > max_depth)).astype(np.float32)
    return hole_mask


def compute_value_weight_online(
    real_depth_m, 
    sim_depth_m, 
    hole_mask,
    sigma=0.1,
    min_weight=0.1
):
    """
    Compute value noise weight on-the-fly (if not pre-computed)
    
    Args:
        real_depth_m: Real depth in meters
        sim_depth_m: Sim depth in meters
        hole_mask: Hole mask (1=hole)
        sigma: Normalization scale
        min_weight: Minimum weight for valid regions
    
    Returns:
        value_weight: Weight in [0, 1]
    """
    diff = np.abs(real_depth_m - sim_depth_m)
    normalized_diff = 1.0 - np.exp(-diff / sigma)
    value_weight = min_weight + (1.0 - min_weight) * normalized_diff
    value_weight = value_weight * (1.0 - hole_mask)
    return value_weight.astype(np.float32)


# ============================================================================
# Dataset Class
# ============================================================================

class ByteCamDepthDataset(Dataset):
    """
    Dual-Modal Noise-Aware Depth Dataset
    
    Supports two modes:
    1. Pre-computed masks: Load from preprocessed directory
    2. Online computation: Compute masks on-the-fly
    
    Directory Structure (Pre-computed):
        data_root/
            rgb/                    # RGB images
            sim_depth/              # Clean simulation depth
            real_depth/             # Noisy real depth
            processed/
                hole_mask/          # Pre-computed hole masks
                value_weight/       # Pre-computed value weights
    
    OR Split-based Structure:
        data_root/
            train.lst               # List of sample IDs
            {sample_id}_rgb.png
            {sample_id}_scan.png    # Real/noisy depth
            {sample_id}_gt.png      # Sim/clean depth
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        resolution: int = 512,
        augment: bool = True,
        
        # Directory names (for directory-based structure)
        rgb_dir: str = 'rgb',
        sim_depth_dir: str = 'sim_depth',
        real_depth_dir: str = 'real_depth',
        processed_dir: str = 'processed',
        
        # Depth configuration
        depth_scale: float = 1000.0,
        d_min: float = 0.05,
        d_max: float = 5.0,
        
        # Noise weight configuration
        sigma: float = 0.1,
        min_weight: float = 0.1,
        
        # Whether to use pre-computed masks
        use_precomputed_masks: bool = True,
        augmentation_mode: str = 'flip_only',
        
        # Structure type
        structure: str = 'auto',  # 'auto', 'lasa', 'directory'
    ):
        """
        Args:
            data_root: Root directory of dataset
            split: 'train', 'val', or 'test'
            resolution: Target image resolution
            augment: Apply data augmentation
            rgb_dir: Subdirectory for RGB images
            sim_depth_dir: Subdirectory for sim depth
            real_depth_dir: Subdirectory for real depth
            processed_dir: Subdirectory for preprocessed masks
            depth_scale: Scale to convert depth to meters
            d_min: Minimum depth for normalization
            d_max: Maximum depth for normalization
            sigma: Sigma for value weight computation
            min_weight: Minimum weight for valid regions
            use_precomputed_masks: Whether to use pre-computed masks
            structure: Dataset structure ('auto', 'lasa', 'directory')
        """
        super().__init__()
        
        self.data_root = data_root
        self.split = split
        self.resolution = resolution
        self.augment = augment
        self.depth_scale = depth_scale
        self.d_min = d_min
        self.d_max = d_max
        self.sigma = sigma
        self.min_weight = min_weight
        self.use_precomputed_masks = use_precomputed_masks
        self.augmentation_mode = augmentation_mode
        
        # Detect structure type
        if structure == 'auto':
            self.structure = self._detect_structure()
        else:
            self.structure = structure
        
        print(f"Dataset structure detected: {self.structure}")
        
        # Load file list based on structure
        if self.structure == 'lasa':
            self._init_lasa_structure()
        else:
            self._init_directory_structure(rgb_dir, sim_depth_dir, real_depth_dir, processed_dir)
        
        print(f"Loaded {len(self.samples)} samples from {split} split")
        
        # Detailed mask status
        if self.use_precomputed_masks:
            if self.has_precomputed:
                print(f"Pre-computed masks: ENABLED ✓")
            else:
                expected_path = os.path.join(self.data_root, processed_dir, 'hole_mask')
                print(f"Pre-computed masks: NOT FOUND!")
                print(f"  Expected path: {expected_path}")
                print(f"  Masks will be computed on-the-fly (slower)")
        else:
            print(f"Pre-computed masks: DISABLED (use_precomputed_masks=False)")
        
        # Transforms
        self.to_tensor = T.ToTensor()
    
    def _detect_structure(self) -> str:
        """Detect dataset structure type"""
        # Check for LASA-style split files
        if os.path.exists(os.path.join(self.data_root, f'{self.split}.lst')):
            return 'lasa'
        # Check for directory-based structure
        elif os.path.isdir(os.path.join(self.data_root, 'rgb')):
            return 'directory'
        else:
            raise ValueError(f"Cannot detect dataset structure in {self.data_root}")
    
    def _init_lasa_structure(self):
        """Initialize for LASA-style structure"""
        split_file = os.path.join(self.data_root, f'{self.split}.lst')
        with open(split_file, 'r') as f:
            sample_ids = [line.strip() for line in f.readlines()]
        
        self.samples = []
        for sid in sample_ids:
            self.samples.append({
                'id': sid,
                'rgb': os.path.join(self.data_root, f'{sid}_rgb.png'),
                'sim_depth': os.path.join(self.data_root, f'{sid}_gt.png'),
                'real_depth': os.path.join(self.data_root, f'{sid}_scan.png'),
            })
        
        # Check for pre-computed masks
        processed_dir = os.path.join(self.data_root, 'processed')
        self.has_precomputed = os.path.isdir(os.path.join(processed_dir, 'hole_mask'))
        
        if self.has_precomputed:
            for sample in self.samples:
                sid = sample['id']
                sample['hole_mask'] = os.path.join(processed_dir, 'hole_mask', f'{sid}_scan_hole.png')
                sample['value_weight'] = os.path.join(processed_dir, 'value_weight', f'{sid}_scan_weight.png')
    
    def _init_directory_structure(self, rgb_dir, sim_depth_dir, real_depth_dir, processed_dir):
        """Initialize for directory-based structure"""
        rgb_path = os.path.join(self.data_root, rgb_dir)
        
        # Get list of RGB files
        rgb_files = sorted([f for f in os.listdir(rgb_path) if f.endswith(('.png', '.jpg', '.PNG', '.JPG'))])
        
        self.samples = []
        for rgb_file in rgb_files:
            stem = os.path.splitext(rgb_file)[0]
            
            sample = {
                'id': stem,
                'rgb': os.path.join(self.data_root, rgb_dir, rgb_file),
                'sim_depth': os.path.join(self.data_root, sim_depth_dir, f'{stem}.png'),
                'real_depth': os.path.join(self.data_root, real_depth_dir, f'{stem}.png'),
            }
            
            # Only add if real_depth exists
            if os.path.exists(sample['real_depth']):
                self.samples.append(sample)
        
        # Check for pre-computed masks
        processed_path = os.path.join(self.data_root, processed_dir)
        self.has_precomputed = os.path.isdir(os.path.join(processed_path, 'hole_mask'))
        
        if self.has_precomputed:
            for sample in self.samples:
                stem = sample['id']
                sample['hole_mask'] = os.path.join(processed_path, 'hole_mask', f'{stem}_hole.png')
                sample['value_weight'] = os.path.join(processed_path, 'value_weight', f'{stem}_weight.png')
    
    def __len__(self):
        return len(self.samples)
    
    def load_image(self, path: str) -> torch.Tensor:
        """Load RGB image as tensor [3, H, W]"""
        img = Image.open(path).convert('RGB')
        return self.to_tensor(img)
    
    def load_depth(self, path: str) -> np.ndarray:
        """Load depth image in meters"""
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_m = depth / self.depth_scale
        return depth_m
    
    def load_mask(self, path: str) -> np.ndarray:
        """Load pre-computed mask (grayscale PNG normalized to [0, 1])"""
        if os.path.exists(path):
            mask = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            mask = mask / 255.0  # Normalize to [0, 1]
            return mask
        else:
            return None
    
    def preprocess_depth(self, depth_m: np.ndarray) -> torch.Tensor:
        """Normalize depth to [0, 1] range"""
        depth_norm = log_scale_depth(depth_m, self.d_min, self.d_max)
        return torch.from_numpy(depth_norm).unsqueeze(0).float()
    
    def apply_augmentation(
        self, 
        rgb: torch.Tensor, 
        sim_depth: torch.Tensor, 
        real_depth: torch.Tensor,
        hole_mask: torch.Tensor,
        value_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Apply joint augmentation to all inputs.

        Supported modes:
        - full: original flip + random resized crop
        - flip_only: preserve tiny structures, only horizontal flip + resize
        - none: deterministic resize only
        """
        mode = self.augmentation_mode if self.augment else 'none'

        if mode in ('flip_only', 'full'):
            do_flip = random.random() > 0.5
            if do_flip:
                rgb = TF.hflip(rgb)
                sim_depth = TF.hflip(sim_depth)
                real_depth = TF.hflip(real_depth)
                hole_mask = TF.hflip(hole_mask)
                value_weight = TF.hflip(value_weight)

        if mode == 'full':
            i, j, h, w = T.RandomResizedCrop.get_params(
                rgb, scale=(0.5, 1.0), ratio=(0.75, 1.33)
            )
            target_size = [self.resolution, self.resolution]
            rgb = TF.resized_crop(
                rgb, i, j, h, w, target_size,
                interpolation=TF.InterpolationMode.BILINEAR
            )
            sim_depth = TF.resized_crop(
                sim_depth, i, j, h, w, target_size,
                interpolation=TF.InterpolationMode.BILINEAR
            )
            real_depth = TF.resized_crop(
                real_depth, i, j, h, w, target_size,
                interpolation=TF.InterpolationMode.BILINEAR
            )
            hole_mask = TF.resized_crop(
                hole_mask, i, j, h, w, target_size,
                interpolation=TF.InterpolationMode.NEAREST
            )
            value_weight = TF.resized_crop(
                value_weight, i, j, h, w, target_size,
                interpolation=TF.InterpolationMode.BILINEAR
            )
        else:
            rgb = self.resize_to_resolution(rgb, interpolation='bilinear')
            sim_depth = self.resize_to_resolution(sim_depth, interpolation='bilinear')
            real_depth = self.resize_to_resolution(real_depth, interpolation='bilinear')
            hole_mask = self.resize_to_resolution(hole_mask, interpolation='nearest')
            value_weight = self.resize_to_resolution(value_weight, interpolation='bilinear')

        hole_mask = (hole_mask > 0.5).float()
        return rgb, sim_depth, real_depth, hole_mask, value_weight
    
    def resize_to_resolution(
        self, 
        tensor: torch.Tensor, 
        interpolation: str = 'bilinear'
    ) -> torch.Tensor:
        """
        Resize tensor to target resolution.
        
        Args:
            tensor: Input tensor [C, H, W] or [H, W]
            interpolation: 'bilinear' for continuous, 'nearest' for binary masks
        """
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        
        mode = TF.InterpolationMode.BILINEAR if interpolation == 'bilinear' else TF.InterpolationMode.NEAREST
        return TF.resize(tensor, [self.resolution, self.resolution], interpolation=mode)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample
        
        Returns:
            sample: Dictionary containing:
                - 'rgb': [3, H, W] RGB image (normalized to [0, 1])
                - 'sim_depth': [1, H, W] Clean/sim depth (normalized)
                - 'real_depth': [1, H, W] Noisy/real depth (normalized)
                - 'hole_mask': [1, H, W] Hole noise mask (binary)
                - 'value_weight': [1, H, W] Value noise weight (continuous)
                - 'training_weight': [1, H, W] Final weight for Latent Branch
                - 'sample_id': Sample ID string
        """
        sample_info = self.samples[idx]
        
        # Load RGB
        rgb = self.load_image(sample_info['rgb'])
        
        # Load depths
        real_depth_m = self.load_depth(sample_info['real_depth'])
        
        # Load sim depth if available
        if os.path.exists(sample_info['sim_depth']):
            sim_depth_m = self.load_depth(sample_info['sim_depth'])
        else:
            # Fallback: use real_depth as sim_depth
            sim_depth_m = real_depth_m.copy()
        
        # Load or compute masks
        if self.use_precomputed_masks and self.has_precomputed:
            hole_mask = self.load_mask(sample_info.get('hole_mask', ''))
            value_weight = self.load_mask(sample_info.get('value_weight', ''))
            
            # Fallback to online computation if loading fails
            if hole_mask is None:
                hole_mask = compute_hole_mask_online(real_depth_m)
            if value_weight is None:
                hole_mask_temp = compute_hole_mask_online(real_depth_m)
                value_weight = compute_value_weight_online(
                    real_depth_m, sim_depth_m, hole_mask_temp, self.sigma, self.min_weight
                )
        else:
            # Compute masks on-the-fly
            hole_mask = compute_hole_mask_online(real_depth_m)
            value_weight = compute_value_weight_online(
                real_depth_m, sim_depth_m, hole_mask, self.sigma, self.min_weight
            )
        
        # Preprocess depths
        sim_depth = self.preprocess_depth(sim_depth_m)
        real_depth = self.preprocess_depth(real_depth_m)
        
        # Convert masks to tensors
        hole_mask = torch.from_numpy(hole_mask).unsqueeze(0).float()
        value_weight = torch.from_numpy(value_weight).unsqueeze(0).float()
        
        # Apply augmentation or resize
        if self.augment or self.augmentation_mode != 'none':
            rgb, sim_depth, real_depth, hole_mask, value_weight = self.apply_augmentation(
                rgb, sim_depth, real_depth, hole_mask, value_weight
            )
        else:
            # Non-augmentation mode: use appropriate interpolation
            rgb = self.resize_to_resolution(rgb, interpolation='bilinear')
            sim_depth = self.resize_to_resolution(sim_depth, interpolation='bilinear')
            real_depth = self.resize_to_resolution(real_depth, interpolation='bilinear')
            # Binary mask uses NEAREST interpolation to preserve sharp edges!
            hole_mask = self.resize_to_resolution(hole_mask, interpolation='nearest')
            value_weight = self.resize_to_resolution(value_weight, interpolation='bilinear')
        
        # ========== Post-processing ==========
        # Clamp continuous values to valid range
        value_weight = value_weight.clamp(0.0, 1.0)
        sim_depth = sim_depth.clamp(0.0, 1.0)
        real_depth = real_depth.clamp(0.0, 1.0)
        
        # CRITICAL: Re-binarize hole_mask to ensure strict 0/1 values
        # This fixes edge blurring caused by any interpolation artifacts
        hole_mask = (hole_mask > 0.5).float()
        
        # Compute final training weight for Latent Branch
        # Hole regions: weight = 0 (ignore)
        # Valid regions: weight = value_weight (focus on high-difference areas)
        training_weight = value_weight * (1.0 - hole_mask)
        
        return {
            'rgb': rgb,
            'sim_depth': sim_depth,
            'real_depth': real_depth,
            'hole_mask': hole_mask,
            'value_weight': value_weight,
            'training_weight': training_weight,
            'sample_id': sample_info['id'],
        }


def create_dataloader(
    data_root: str,
    split: str = 'train',
    batch_size: int = 4,
    resolution: int = 512,
    num_workers: int = 4,
    augment: bool = True,
    shuffle: bool = True,
    use_precomputed_masks: bool = True,
    augmentation_mode: str = 'flip_only',
    **kwargs
):
    """
    Create DataLoader for training/validation
    """
    dataset = ByteCamDepthDataset(
        data_root=data_root,
        split=split,
        resolution=resolution,
        augment=augment,
        use_precomputed_masks=use_precomputed_masks,
        augmentation_mode=augmentation_mode,
        **kwargs
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader


DualModalNoiseDataset = ByteCamDepthDataset
