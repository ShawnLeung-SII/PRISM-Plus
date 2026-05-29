"""
Vision Foundation Model (VFM) Interface v2

Supports multiple VFM backends:
  - MoGe2: From pixel-perfect-depth, uses DINOv2-Large backbone internally
  - DINOv2: Direct use of DINOv2 models from PyTorch Hub

Key Features:
  - Unified interface for semantic feature extraction
  - Support for multi-scale intermediate layer extraction
  - ImageNet normalization handled internally

VFM Comparison:
┌──────────┬────────────────┬─────────────┬────────────┬──────────────────┐
│ Model    │ Backbone       │ Feature Dim │ Patch Size │ Pre-trained Data │
├──────────┼────────────────┼─────────────┼────────────┼──────────────────┤
│ MoGe2    │ DINOv2-Large   │ 1024        │ 14         │ MoGe dataset     │
│ DINOv2-L │ ViT-Large      │ 1024        │ 14         │ LVD-142M         │
│ DINOv2-G │ ViT-Giant      │ 1536        │ 14         │ LVD-142M         │
└──────────┴────────────────┴─────────────┴────────────┴──────────────────┘

Recommended:
  - MoGe2: Best for depth-related tasks (pre-trained on depth data)
  - DINOv2-L: General purpose, good balance of speed and quality

Weight Download:
  - MoGe2: Auto-downloads from HuggingFace (Ruicheng/moge-vitl)
  - DINOv2: Auto-downloads from PyTorch Hub
  - No manual download required!
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union, Tuple
from abc import ABC, abstractmethod


class VFMInterface(ABC, nn.Module):
    """Abstract base class for Vision Foundation Models"""
    
    @abstractmethod
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Extract semantic features from RGB images
        
        Args:
            rgb: [B, 3, H, W] RGB image in [0, 1] or ImageNet normalized
            
        Returns:
            features: [B, N, C] semantic features (N = num_patches)
        """
        pass
    
    @abstractmethod
    def get_intermediate_layers(
        self, 
        rgb: torch.Tensor, 
        n: Union[int, List[int]] = 4,
        return_class_token: bool = False,
    ) -> List[torch.Tensor]:
        """
        Extract intermediate layer features for multi-scale decoding
        
        Args:
            rgb: [B, 3, H, W] RGB image
            n: Number of layers or list of layer indices
            return_class_token: Whether to return class token
            
        Returns:
            List of [B, N, C] tensors from intermediate layers
        """
        pass
    
    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Return embedding dimension"""
        pass
    
    @property
    @abstractmethod
    def patch_size(self) -> int:
        """Return patch size (e.g., 14)"""
        pass


class MoGe2VFM(VFMInterface):
    """
    MoGe2 Vision Foundation Model
    
    From: Pixel-Perfect-Depth (https://github.com/prs-eth/pixel-perfect-depth)
    
    Architecture: DINOv2-Large backbone + MoGe-specific training
    Feature: 1024-dim at 1/14 scale
    
    Key Advantage: Pre-trained on depth estimation task, better geometric priors
    
    Weight Download:
        The weights will be automatically downloaded from HuggingFace:
        - Repo: Ruicheng/moge-vitl
        - File: model.pt
        
    Alternatively, you can manually download:
        1. From HuggingFace: https://huggingface.co/Ruicheng/moge-vitl
        2. Place at: weights/moge2.pt (or auto-download from HuggingFace)
    """
    
    def __init__(
        self, 
        checkpoint_path: Optional[str] = None,
        freeze: bool = True,
    ):
        """
        Args:
            checkpoint_path: Path to MoGe2 checkpoint. If None, downloads from HuggingFace.
            freeze: Whether to freeze parameters
        """
        super().__init__()
        
        # Import MoGe2 model
        try:
            from prism_plus.models.vfm.moge.model.v2 import MoGeModel
        except ImportError:
            raise ImportError(
                "MoGe2 not found. Please ensure the ppd/moge directory exists.\n"
                "The MoGe2 code should be included in prism_plus/models/vfm/moge/"
            )
        
        # Load model - from_pretrained handles HuggingFace download automatically
        if checkpoint_path is None:
            # Default: download from HuggingFace
            print("Loading MoGe2 from HuggingFace: Ruicheng/moge-vitl")
            self.model = MoGeModel.from_pretrained("Ruicheng/moge-vitl")
        else:
            print(f"Loading MoGe2 from local: {checkpoint_path}")
            self.model = MoGeModel.from_pretrained(checkpoint_path)
        
        # Get embedding dimension from the encoder
        self._embed_dim = self.model.encoder.dim_features  # Should be 1024 for ViT-L
        # MoGe2 uses patch_size=16 internally in forward_semantics (see moge/model/v2.py)
        self._patch_size = 16
        
        # Configure intermediate layers for multi-scale extraction
        # DINOv2-Large has 24 blocks, we extract from blocks [4, 11, 17, 23]
        self._intermediate_layers = [4, 11, 17, 23]
        
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            print("MoGe2 VFM frozen (not trainable)")
        
        print(f"MoGe2 VFM initialized: embed_dim={self._embed_dim}, patch_size={self._patch_size}")
    
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Extract MoGe2 features
        
        Args:
            rgb: [B, 3, H, W] in [0, 1]
            
        Returns:
            features: [B, N, C] where N = (H//16) * (W//16), C = 1024
        """
        # MoGe2's forward_semantics returns [B, N, C] format
        features = self.model.forward_semantics(rgb)
        return features
    
    def get_intermediate_layers(
        self, 
        rgb: torch.Tensor, 
        n: Union[int, List[int]] = 4,
        return_class_token: bool = False,
    ) -> List[torch.Tensor]:
        """
        Extract intermediate layer features from MoGe2's DINOv2 backbone
        
        Args:
            rgb: [B, 3, H, W] in [0, 1]
            n: Number of layers (uses default indices) or list of layer indices
            
        Returns:
            List of [B, N, C] tensors
        """
        B, _, H, W = rgb.shape
        base_h, base_w = H // self._patch_size, W // self._patch_size
        
        # Determine which layers to extract
        if isinstance(n, int):
            # Use default layer indices
            layer_indices = self._intermediate_layers[:n]
        else:
            layer_indices = n
        
        # Access the DINOv2 backbone inside MoGe2
        # The backbone is self.model.encoder.backbone (DinoVisionTransformer)
        backbone = self.model.encoder.backbone
        
        # Normalize input (MoGe2's encoder does this internally)
        mean = self.model.encoder.image_mean.to(rgb.device)
        std = self.model.encoder.image_std.to(rgb.device)
        x = (rgb - mean) / std
        
        # Get intermediate features using DINOv2's get_intermediate_layers
        features = backbone.get_intermediate_layers(
            x, 
            n=layer_indices,
            reshape=False,  # Keep as [B, N, C]
            return_class_token=return_class_token,
        )
        
        return list(features)
    
    @property
    def embed_dim(self) -> int:
        return self._embed_dim
    
    @property
    def patch_size(self) -> int:
        return self._patch_size


class DINOv2VFM(VFMInterface):
    """
    DINOv2 Vision Foundation Model (Direct Use)
    
    From: PyTorch Hub (facebookresearch/dinov2)
    
    Available models:
        - dinov2_vits14: ViT-Small (384 dim, fastest)
        - dinov2_vitb14: ViT-Base (768 dim)
        - dinov2_vitl14: ViT-Large (1024 dim, recommended)
        - dinov2_vitg14: ViT-Giant (1536 dim, highest quality)
    
    Weight Download:
        Weights are automatically downloaded from PyTorch Hub.
        No manual download needed!
        
        Cache location: ~/.cache/torch/hub/checkpoints/
    """
    
    # Embedding dimensions for each model variant
    EMBED_DIMS = {
        'dinov2_vits14': 384,
        'dinov2_vitb14': 768,
        'dinov2_vitl14': 1024,
        'dinov2_vitg14': 1536,
    }
    
    # Intermediate layer indices for multi-scale extraction
    LAYER_INDICES = {
        'dinov2_vits14': [2, 5, 8, 11],   # 12 blocks
        'dinov2_vitb14': [2, 5, 8, 11],   # 12 blocks
        'dinov2_vitl14': [4, 11, 17, 23], # 24 blocks
        'dinov2_vitg14': [9, 19, 29, 39], # 40 blocks
    }
    
    def __init__(
        self,
        model_name: str = 'dinov2_vitl14',
        freeze: bool = True,
    ):
        """
        Args:
            model_name: DINOv2 model name
            freeze: Whether to freeze parameters
        """
        super().__init__()
        
        if model_name not in self.EMBED_DIMS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.EMBED_DIMS.keys())}")
        
        self._model_name = model_name
        self._embed_dim = self.EMBED_DIMS[model_name]
        self._patch_size = 14
        self._intermediate_layers = self.LAYER_INDICES[model_name]
        
        # Load from PyTorch Hub (auto-download)
        print(f"Loading DINOv2 from PyTorch Hub: facebookresearch/dinov2/{model_name}")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        
        # ImageNet normalization
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            print("DINOv2 VFM frozen (not trainable)")
        
        print(f"DINOv2 VFM initialized: {model_name}, embed_dim={self._embed_dim}")
    
    def _normalize(self, rgb: torch.Tensor) -> torch.Tensor:
        """Normalize RGB from [0,1] to ImageNet normalized"""
        return (rgb - self.image_mean.to(rgb.device)) / self.image_std.to(rgb.device)
    
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Extract DINOv2 features
        
        Args:
            rgb: [B, 3, H, W] in [0, 1]
            
        Returns:
            features: [B, N, C] patch tokens
        """
        x = self._normalize(rgb)
        
        # Get patch tokens (exclude class token)
        features = self.model.forward_features(x)
        patch_tokens = features['x_norm_patchtokens']  # [B, N, C]
        
        return patch_tokens
    
    def get_intermediate_layers(
        self, 
        rgb: torch.Tensor, 
        n: Union[int, List[int]] = 4,
        return_class_token: bool = False,
    ) -> List[torch.Tensor]:
        """
        Extract intermediate layer features
        
        Args:
            rgb: [B, 3, H, W] in [0, 1]
            n: Number of layers or list of layer indices
            
        Returns:
            List of [B, N, C] tensors
        """
        x = self._normalize(rgb)
        
        # Determine which layers to extract
        if isinstance(n, int):
            layer_indices = self._intermediate_layers[:n]
        else:
            layer_indices = n
        
        # Use DINOv2's built-in method
        features = self.model.get_intermediate_layers(
            x,
            n=layer_indices,
            reshape=False,
            return_class_token=return_class_token,
        )
        
        return list(features)
    
    @property
    def embed_dim(self) -> int:
        return self._embed_dim
    
    @property
    def patch_size(self) -> int:
        return self._patch_size


def create_vfm(
    model_type: str = 'moge2',
    checkpoint_path: Optional[str] = None,
    freeze: bool = True,
    **kwargs
) -> VFMInterface:
    """
    Factory function to create VFM
    
    Args:
        model_type: Type of VFM
            - 'moge2': MoGe2 model (recommended for depth tasks)
            - 'dinov2': DINOv2 model (general purpose)
        checkpoint_path: Path to checkpoint (for MoGe2, optional - auto-downloads)
        freeze: Whether to freeze parameters
        **kwargs: Additional arguments
        
    Returns:
        vfm: VFM instance
        
    Examples:
        # MoGe2 - auto-download from HuggingFace
        >>> vfm = create_vfm('moge2')
        
        # MoGe2 - with local checkpoint
        >>> vfm = create_vfm('moge2', checkpoint_path='checkpoints/moge2/model.pt')
        
        # DINOv2-Large - auto-download from PyTorch Hub
        >>> vfm = create_vfm('dinov2', model_name='dinov2_vitl14')
    """
    model_type = model_type.lower()
    
    if model_type == 'moge2':
        return MoGe2VFM(
            checkpoint_path=checkpoint_path,
            freeze=freeze,
        )
    
    elif model_type in ['dinov2', 'dino']:
        model_name = kwargs.get('model_name', 'dinov2_vitl14')
        return DINOv2VFM(
            model_name=model_name,
            freeze=freeze,
        )
    
    else:
        raise ValueError(
            f"Unknown VFM type: {model_type}. "
            f"Supported: 'moge2', 'dinov2'"
        )


class VFMProjector(nn.Module):
    """
    Projects VFM features to target dimension
    
    Used to adapt VFM features to different branch requirements
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 2
    ):
        """
        Args:
            input_dim: VFM feature dimension
            output_dim: Target feature dimension
            num_layers: Number of projection layers
        """
        super().__init__()
        
        if num_layers == 1:
            self.projector = nn.Linear(input_dim, output_dim)
        else:
            layers = []
            hidden_dim = (input_dim + output_dim) // 2
            
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            
            layers.append(nn.Linear(hidden_dim, output_dim))
            
            self.projector = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Project features
        
        Args:
            features: [B, N, C_in] or [B, C_in, H, W]
            
        Returns:
            projected: [B, N, C_out] or [B, C_out, H, W]
        """
        return self.projector(features)
