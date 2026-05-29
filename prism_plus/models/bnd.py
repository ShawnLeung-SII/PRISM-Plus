"""PRISM Binary Noise Discriminator (BND).

This module implements the BND backbone described in:
    "PRISM: Learning Realistic Depth via Physics-Grounded Noise
     Disentanglement with Semantic-Geometric Collaboration" (ICML 2026).

Architecture summary:
    RGB+Depth → PixelLevelEncoder → UNetDecoder + VFM GlobalSemanticContext
              → failure_head + detail_residual_head → invalidation mask

This is the BASELINE BND (PRISM ICML). For the PRISM+ Spatial-SPR variant,
see :class:`prism_plus.models.bnd_spatial.SpatialBND`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Literal

# 兼容 vfm_interface v1 和 v2
try:
    from .vfm import create_vfm, VFMInterface
except ImportError:
    from .vfm import create_vfm  # legacy fallback
    VFMInterface = nn.Module  # 兼容类型注解


# ============================================================================
# 配置
# ============================================================================

ENCODER_CONFIGS = {
    'small': {
        'vfm_dim': 384,
        'encoder_dims': [32, 64, 128, 256],
        'decoder_dim': 64,
    },
    'base': {
        'vfm_dim': 768,
        'encoder_dims': [64, 128, 256, 512],
        'decoder_dim': 128,
    },
    'large': {
        'vfm_dim': 1024,
        'encoder_dims': [64, 128, 256, 512],
        'decoder_dim': 128,
    },
}


# ============================================================================
# 模块 1: Pixel-Level Encoder (纯卷积，保持像素信息)
# ============================================================================

class ConvBlock(nn.Module):
    """标准卷积块: Conv-BN-ReLU"""
    
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        
        # 残差连接
        self.skip = nn.Identity() if (in_ch == out_ch and stride == 1) else \
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.skip(x)


class GeometryFeatureExtractor(nn.Module):
    """从 sim depth 中提取局部几何导数特征"""

    OUT_CHANNELS = 5

    def __init__(self):
        super().__init__()
        self.register_buffer(
            'sobel_x',
            torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        )
        self.register_buffer(
            'sobel_y',
            torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        )
        self.register_buffer(
            'laplacian',
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        grad_x = F.conv2d(depth, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        lap = F.conv2d(depth, self.laplacian, padding=1)
        return torch.cat([depth, grad_x, grad_y, grad_mag, lap], dim=1)


class PixelLevelEncoder(nn.Module):
    """
    像素级编码器 - 纯卷积设计
    
    输入: RGB [B, 3, H, W] + Geometry(depth) [B, 5, H, W]
    输出: 多尺度特征 [F0(H), F1(H/2), F2(H/4), F3(H/8), F4(H/16)]
    
    关键改进 (V5.1):
    - 新增 F0: 全分辨率特征，用于 Decoder 最后一步
    - 接通"生命线"，细节不再丢失
    """
    
    # 全分辨率特征通道数 (用于 F0 和 Dec1 融合)
    F0_CHANNELS = 32
    STEM_CHANNELS = 16
    
    def __init__(self, dims: List[int] = [64, 128, 256, 512]):
        super().__init__()
        
        self.dims = dims
        self.geometry_extractor = GeometryFeatureExtractor()

        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, self.STEM_CHANNELS, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.STEM_CHANNELS),
            nn.ReLU(inplace=True),
        )
        self.geometry_stem = nn.Sequential(
            nn.Conv2d(self.geometry_extractor.OUT_CHANNELS, self.STEM_CHANNELS, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.STEM_CHANNELS),
            nn.ReLU(inplace=True),
        )

        self.pre_stem = nn.Sequential(
            nn.Conv2d(self.STEM_CHANNELS * 2, self.F0_CHANNELS, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.F0_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.F0_CHANNELS, self.F0_CHANNELS, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        
        # ============ 原有的 Stem (H -> H/2) ============
        # 注意: input channel 变成了 F0_CHANNELS (32)
        self.stem = nn.Sequential(
            nn.Conv2d(self.F0_CHANNELS, dims[0], 3, 2, 1, bias=False),  # Stride=2 下采样
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True),
        )
        
        # Stage 1: H/2 → H/2
        self.stage1 = nn.Sequential(
            ConvBlock(dims[0], dims[0]),
            ConvBlock(dims[0], dims[0]),
        )
        
        # Stage 2: H/2 → H/4
        self.stage2 = nn.Sequential(
            ConvBlock(dims[0], dims[1], stride=2),
            ConvBlock(dims[1], dims[1]),
        )
        
        # Stage 3: H/4 → H/8
        self.stage3 = nn.Sequential(
            ConvBlock(dims[1], dims[2], stride=2),
            ConvBlock(dims[2], dims[2]),
        )
        
        # Stage 4: H/8 → H/16
        self.stage4 = nn.Sequential(
            ConvBlock(dims[2], dims[3], stride=2),
            ConvBlock(dims[3], dims[3]),
        )
    
    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict:
                - features: [F0, F1, F2, F3, F4]
                - rgb0: RGB 浅层特征
                - geo0: Geometry 浅层特征
                - geometry_maps: 几何导数图
        """
        geometry_maps = self.geometry_extractor(depth)
        rgb0 = self.rgb_stem(rgb)
        geo0 = self.geometry_stem(geometry_maps)
        x = torch.cat([rgb0, geo0], dim=1)

        f0 = self.pre_stem(x)
        x = self.stem(f0)      # [B, 64, H/2, W/2]
        
        f1 = self.stage1(x)    # H/2
        f2 = self.stage2(f1)   # H/4
        f3 = self.stage3(f2)   # H/8
        f4 = self.stage4(f3)   # H/16
        
        return {
            'features': [f0, f1, f2, f3, f4],
            'rgb0': rgb0,
            'geo0': geo0,
            'geometry_maps': geometry_maps,
        }


# ============================================================================
# 模块 2: VFM Semantic Guidance (全局语义 Context，不是 Spatial Attention!)
# ============================================================================

class GlobalSemanticContext(nn.Module):
    """
    VFM 全局语义上下文模块 (重新设计 - 避免块状问题)
    
    ==================== 关键改进 ====================
    
    旧设计的问题:
    - VFM 输出 [B, N, C] 是 patch-level 的 (每个 token = 14×14 像素)
    - 上采样到像素级只是插值，不能创造边界信息
    - Spatial attention 把块状 pattern 污染到 CNN 特征
    
    新设计:
    - VFM 输出做 Global Average Pooling → [B, C] 全局向量
    - 用这个向量生成 Channel-wise attention (类似 SE-Net)
    - ✅ 不引入空间维度的块状 pattern
    - ✅ 保持 CNN 的像素级边界信息
    
    语义信息如何帮助:
    - "这张图有玻璃" → 增强 failure 检测 channel
    - "这是金属表面" → 增强 uncertainty 估计 channel
    - 全局语义 + 像素级特征 = 精确预测
    """
    
    def __init__(
        self,
        vfm_type: str = 'moge2',
        vfm_checkpoint: Optional[str] = None,
        vfm_dim: int = 1024,
        target_dims: List[int] = [64, 128, 256, 512],
    ):
        super().__init__()
        
        # 加载 VFM
        try:
            self.vfm = create_vfm(
                model_type=vfm_type,
                checkpoint_path=vfm_checkpoint,
                freeze=True,
            )
        except TypeError:
            self.vfm = create_vfm(
                vfm_type=vfm_type,
                checkpoint_path=vfm_checkpoint,
                freeze=True,
            )
        
        # 获取 VFM 维度
        if hasattr(self.vfm, 'embed_dim'):
            self.vfm_dim = self.vfm.embed_dim
        elif hasattr(self.vfm, 'feature_dim'):
            self.vfm_dim = self.vfm.feature_dim
        else:
            self.vfm_dim = vfm_dim
        
        # 全局语义编码器
        self.global_encoder = nn.Sequential(
            nn.Linear(self.vfm_dim, self.vfm_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.vfm_dim // 2, self.vfm_dim // 4),
            nn.ReLU(inplace=True),
        )
        
        # 为每个尺度生成 Channel-wise attention (SE-Net style)
        # 改进：使用 Tanh 输出 [-1, 1]，配合残差调制 skip * (1 + weight)
        # 这样 weight=-1 时衰减，weight=0 时不变，weight=1 时增强
        self.channel_attention_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.vfm_dim // 4, dim),
                nn.Tanh(),  # 输出 [-1, 1]，配合残差调制使用
            )
            for dim in target_dims
        ])
    
    def forward(self, rgb: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            rgb: [B, 3, H, W] - 只需要 RGB，不需要 depth!
        
        Returns:
            channel_weights: List of [B, C_i, 1, 1] channel attention weights
                           用于调制 CNN encoder 的每个尺度特征
        """
        B = rgb.shape[0]
        
        # 1. 获取 VFM 特征
        with torch.no_grad():
            vfm_features = self.vfm(rgb)
            
            # 处理不同的返回格式
            if isinstance(vfm_features, dict):
                if 'x_norm_patchtokens' in vfm_features:
                    vfm_feat = vfm_features['x_norm_patchtokens']
                elif 'last_hidden_state' in vfm_features:
                    vfm_feat = vfm_features['last_hidden_state']
                else:
                    vfm_feat = list(vfm_features.values())[0]
            elif isinstance(vfm_features, list):
                vfm_feat = vfm_features[-1]
            else:
                vfm_feat = vfm_features
        
        # 2. Global Average Pooling → [B, C]
        # 关键：消除空间维度，避免块状 pattern
        if vfm_feat.dim() == 3:  # [B, N, C]
            global_feat = vfm_feat.mean(dim=1)  # [B, C]
        elif vfm_feat.dim() == 4:  # [B, C, H, W]
            global_feat = vfm_feat.mean(dim=[2, 3])  # [B, C]
        else:
            raise RuntimeError(f"Unexpected VFM output shape: {vfm_feat.shape}")
        
        # 3. 编码全局语义
        global_context = self.global_encoder(global_feat)  # [B, C/4]
        
        # 4. 为每个尺度生成 channel attention
        channel_weights = []
        for attn_gen in self.channel_attention_generators:
            weight = attn_gen(global_context)  # [B, C_i]
            weight = weight.unsqueeze(-1).unsqueeze(-1)  # [B, C_i, 1, 1]
            channel_weights.append(weight)
        
        return channel_weights


# ============================================================================
# 模块 3: UNet-style Decoder (对称跳连，像素级恢复)
# ============================================================================

class DecoderBlock(nn.Module):
    """
    UNet 解码块: 上采样 + Skip + Conv
    
    硬伤九修复: PixelShuffle 后不用 BatchNorm
    - PixelShuffle 是通道重排操作，它把 4 个通道变成 2x2 的空间位置
    - 紧跟 BN 会在这些刚重排的特征上做归一化，可能破坏像素间的相对关系
    - 改为: PixelShuffle 后直接 ReLU
    """
    
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        
        # 上采样 (使用 PixelShuffle 保持边缘)
        # 硬伤九修复: PixelShuffle 后不用 BN，只用 ReLU
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 4, 3, 1, 1, bias=True),  # bias=True 因为没有 BN
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),  # 直接 ReLU，不要 BN
        )
        
        # 融合 skip connection
        # 硬伤十修复: 第一层用 GN，第二层不用归一化
        self.fusion = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, 1, 1, bias=False),
            nn.GroupNorm(8, out_ch),  # 用 GN 替代 BN
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=True),  # bias=True，不用 BN
            nn.ReLU(inplace=True),  # 不用归一化，保留细节
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        skip: torch.Tensor, 
        channel_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: 上一层特征 [B, C, H, W]
            skip: Encoder 跳连特征 [B, C_skip, 2H, 2W]
            channel_weight: 全局语义 channel attention [B, C_skip, 1, 1] (可选)
        
        修复的问题:
        1. 硬伤一: 插值方式从 nearest 改为 bilinear
           - nearest 会把像素复制成2x2方块，导致粒度粗糙
           - bilinear 插值能保持边缘平滑
           
        2. 硬伤二: channel_weight 从门控改为残差调制
           - 原来: skip * weight → 如果 weight≈0，高频细节被抹杀
           - 现在: skip * (1 + weight) → 原始细节100%保留，VFM只做调制
        """
        # 上采样
        x = self.up(x)
        
        # 硬伤一修复: 使用 bilinear 插值而不是 nearest
        # nearest 会导致方块状伪影，这是粒度粗糙的物理原因
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # 硬伤二修复: Residual Modulation (残差调制)
        # Skip Connection 的作用是保留高频细节(边缘、纹理)
        # 原来的 skip * weight 会把 weight≈0 的区域细节抹杀
        # 
        # 现在使用: skip * (1 + weight)
        # - weight 使用 Tanh 输出 [-1, 1]
        # - weight = -1: 衰减到 0 (极端情况，几乎不会发生)
        # - weight = 0: 保持原样，细节 100% 保留
        # - weight = +1: 增强到 2x
        # 
        # 这样 VFM 的作用是"调制"而不是"门控"
        if channel_weight is not None:
            # channel_weight: [B, C, 1, 1]，值域 [-1, 1]
            skip = skip * (1.0 + channel_weight)  # ✅ 残差调制，保留细节
        
        # 融合
        x = torch.cat([x, skip], dim=1)
        x = self.fusion(x)
        
        return x


class UNetDecoder(nn.Module):
    """
    UNet 风格解码器 (V5.1 - 全分辨率 Skip Connection)
    
    关键改进: 
    - dec1 现在接收 F0 (全分辨率特征) 作为 skip connection
    - 不再是"瞎猜"，而是有原始像素信息作参考
    """
    
    def __init__(self, encoder_dims: List[int], decoder_dim: int):
        super().__init__()
        
        # encoder_dims: [64, 128, 256, 512] for H/2, H/4, H/8, H/16
        # 解码方向: H/16 → H/8 → H/4 → H/2 → H/1
        
        # H/16 → H/8
        self.dec4 = DecoderBlock(encoder_dims[3], encoder_dims[2], decoder_dim * 4)
        
        # H/8 → H/4
        self.dec3 = DecoderBlock(decoder_dim * 4, encoder_dims[1], decoder_dim * 2)
        
        # H/4 → H/2
        self.dec2 = DecoderBlock(decoder_dim * 2, encoder_dims[0], decoder_dim)
        
        # ============ V5.1 关键修改: H/2 → H/1 (有 F0 skip!) ============
        # 原来: 只有 PixelShuffle 上采样，没有 skip connection，"瞎猜"细节
        # 现在: 上采样后与 F0 (全分辨率特征) concat，细节不再丢失!
        
        # Step 1: 上采样 H/2 → H
        self.dec1_up = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim * 4, 3, 1, 1, bias=False),
            nn.PixelShuffle(2),  # H/2 → H
            nn.GroupNorm(8, decoder_dim),
            nn.ReLU(inplace=True),
        )
        
        # Step 2: 融合 F0 skip connection
        # 输入通道: decoder_dim (来自上采样) + F0_CHANNELS (来自 F0, 即 32)
        self.dec1_fusion = nn.Sequential(
            nn.Conv2d(decoder_dim + PixelLevelEncoder.F0_CHANNELS, decoder_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(
        self, 
        encoder_features: List[torch.Tensor],  # [F0, F1, F2, F3, F4] - 注意现在有 5 个!
        channel_weights: Optional[List[torch.Tensor]] = None,  # [W1, W2, W3, W4] channel attention
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            out: [B, decoder_dim, H, W] 最终特征
            pyramid: [P4, P3, P2, P1] 多尺度特征 (用于深度监督)
        """
        # 注意：现在 encoder_features 有 5 个元素!
        f0, f1, f2, f3, f4 = encoder_features  # H, H/2, H/4, H/8, H/16
        
        if channel_weights is not None:
            w1, w2, w3, w4 = channel_weights  # [B, C, 1, 1] channel weights
        else:
            w1, w2, w3, w4 = None, None, None, None
        
        # 解码 (使用 channel-wise attention，不是 spatial attention)
        p4 = f4  # H/16 (起点)
        if w4 is not None:
            p4 = p4 * (1.0 + w4)
        p3 = self.dec4(p4, f3, w3)  # H/8
        p2 = self.dec3(p3, f2, w2)  # H/4
        p1 = self.dec2(p2, f1, w1)  # H/2
        
        # ============ V5.1 关键修改区 ============
        # Step 1: 上采样 p1 到 H
        out_up = self.dec1_up(p1)  # [B, decoder_dim, H, W]
        
        # Step 2: 强制尺寸对齐 (使用切片，严禁 interpolate!)
        if out_up.shape[2:] != f0.shape[2:]:
            h, w = f0.shape[2:]
            out_up = out_up[:, :, :h, :w]
        
        # Step 3: Concat 全分辨率特征 F0
        out = torch.cat([out_up, f0], dim=1)  # [B, decoder_dim + 32, H, W]
        
        # Step 4: 融合
        out = self.dec1_fusion(out)  # [B, decoder_dim, H, W]
        
        pyramid = [p4, p3, p2, p1]
        
        return out, pyramid


# ============================================================================
# 模块 4: Dense Prediction Heads
# ============================================================================

class PixelPredictionLogitHead(nn.Module):
    """
    像素级预测头
    
    硬伤八修复: 去掉最后两层的 BatchNorm
    - BatchNorm 会将特征归一化到均值0、方差1
    - 这会抑制局部的极端值(即孤立的异常点)
    - 对于检测细粒度孤立点，应该保留这些极端响应
    
    改进:
    1. 第一层使用 GroupNorm (比 BN 更尊重空间结构)
    2. 第二层完全不用归一化 (保留细节响应)
    """
    
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.head = nn.Sequential(
            # 第一层: 特征降维，使用 GN (比 BN 更尊重空间结构)
            nn.Conv2d(in_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.GroupNorm(8, hidden_dim),  # 8 groups, 比 BN 更尊重空间结构
            nn.ReLU(inplace=True),
            # 第二层: 细化特征，不用归一化 (保留极端响应)
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True),  # 有 bias，无 BN
            nn.ReLU(inplace=True),
            # 输出层
            nn.Conv2d(hidden_dim, 1, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class DetailResidualHead(nn.Module):
    """全分辨率细节残差分支，只负责补碎点和细边"""

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiScaleHead(nn.Module):
    """多尺度预测头 (用于深度监督)"""
    
    def __init__(self, in_dims: List[int]):
        super().__init__()
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim // 2, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 2, 1, 1),
            )
            for dim in in_dims
        ])
    
    def forward(
        self, 
        pyramid: List[torch.Tensor], 
        target_size: Tuple[int, int],
    ) -> List[torch.Tensor]:
        """输出上采样到 target_size 的预测"""
        # 硬伤六修复: 使用 align_corners=False 保持边界位置正确
        preds = []
        for feat, head in zip(pyramid, self.heads):
            pred = head(feat)
            pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
            preds.append(pred)
        return preds


# ============================================================================
# 主模型: Dual-Stream Pixel Branch V9
# ============================================================================

class BND(nn.Module):
    """
    Dual-Stream Pixel Branch V9 - Pixel-Centric Detail-Enhanced Design
    
    核心改变:
    1. 主干是纯卷积网络，不是 VFM
    2. VFM 只提供语义 attention，不主导预测
    3. UNet 对称跳连，像素级信息完整保持
    4. 预测在全分辨率进行
    """
    
    def __init__(
        self,
        vfm_type: str = 'moge2',
        vfm_checkpoint: Optional[str] = None,
        encoder_size: Literal['small', 'base', 'large'] = 'large',
        use_semantic_guidance: bool = True,
        deep_supervision: bool = True,
    ):
        super().__init__()
        
        config = ENCODER_CONFIGS[encoder_size]
        self.use_semantic_guidance = use_semantic_guidance
        self.deep_supervision = deep_supervision
        
        print("\n" + "="*70)
        print("Initializing Dual-Stream Pixel Branch V9 (Pixel-Centric Detail-Enhanced)")
        print(f"  Encoder: {encoder_size}")
        print(f"  VFM: {vfm_type} (semantic guidance only)")
        print(f"  Semantic Guidance: {use_semantic_guidance}")
        print(f"  Deep Supervision: {deep_supervision}")
        print("="*70)
        
        # 1. Pixel-Level Encoder (主干)
        print("\n[1/4] Creating Pixel-Level Encoder (主干)...")
        self.encoder = PixelLevelEncoder(dims=config['encoder_dims'])
        
        # 2. VFM Global Semantic Context (辅助 - 不是 Spatial Attention!)
        if use_semantic_guidance:
            print("[2/4] Loading VFM Global Semantic Context...")
            print("      ✅ 使用 Channel Attention，不是 Spatial Attention")
            print("      ✅ 避免 VFM patch-level 特征污染像素级信息")
            self.semantic_context = GlobalSemanticContext(
                vfm_type=vfm_type,
                vfm_checkpoint=vfm_checkpoint,
                vfm_dim=config['vfm_dim'],
                target_dims=config['encoder_dims'],
            )
        else:
            self.semantic_context = None
            print("[2/4] Skipping VFM (no semantic guidance)")
        
        # 3. UNet Decoder
        print("[3/4] Creating UNet-style Decoder...")
        self.decoder = UNetDecoder(
            encoder_dims=config['encoder_dims'],
            decoder_dim=config['decoder_dim'],
        )
        
        # 4. Prediction Heads
        print("[4/4] Creating Prediction Heads...")
        self.failure_head = PixelPredictionLogitHead(config['decoder_dim'])
        detail_in_dim = (
            config['decoder_dim']
            + PixelLevelEncoder.F0_CHANNELS
            + PixelLevelEncoder.STEM_CHANNELS * 2
            + GeometryFeatureExtractor.OUT_CHANNELS
            + 1
        )
        self.detail_head = DetailResidualHead(detail_in_dim, hidden_dim=64)
        self.detail_scale = nn.Parameter(torch.tensor(0.5))
        
        # 深度监督头
        if deep_supervision:
            # Decoder pyramid dims: [512, decoder_dim*4, decoder_dim*2, decoder_dim]
            pyramid_dims = [
                config['encoder_dims'][3],  # H/16
                config['decoder_dim'] * 4,  # H/8
                config['decoder_dim'] * 2,  # H/4
                config['decoder_dim'],      # H/2
            ]
            self.aux_failure_heads = MultiScaleHead(pyramid_dims)
        
        # 统计参数
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        if use_semantic_guidance:
            vfm_params = sum(p.numel() for p in self.semantic_context.vfm.parameters())
            print(f"\n  VFM params: {vfm_params:,} (frozen)")
        
        print(f"  Total params:     {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        print("="*70 + "\n")
    
    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            rgb: [B, 3, H, W]
            depth: [B, 1, H, W]
        
        Returns:
            dict with:
                - pred_failure: [B, 1, H, W] Invalidation mask
                - pred_main: [B, 1, H, W] Main branch probability
                - pred_detail: [B, 1, H, W] Detail residual probability
                - aux_failure_maps: List[Tensor] (if deep_supervision)
        """
        B, _, H, W = rgb.shape
        
        # 1. Encoder (像素级特征提取)
        # V5.1: 现在返回 [F0, F1, F2, F3, F4]，其中 F0 是全分辨率特征
        encoder_out = self.encoder(rgb, depth)
        encoder_features = encoder_out['features']
        f0 = encoder_features[0]
        
        # 2. VFM Global Semantic Context (可选)
        # ✅ 只提供 channel-wise 权重，不影响空间结构
        # 注意: VFM 只调制 F1-F4，不调制 F0 (保持原始细节)
        if self.semantic_context is not None:
            channel_weights = self.semantic_context(rgb)  # [B, C_i, 1, 1] list (4个)
        else:
            channel_weights = None
        
        # 3. Decoder (像素级恢复，使用 channel attention)
        # V5.1: decoder 现在会使用 F0 作为最后一步的 skip connection
        out, pyramid = self.decoder(encoder_features, channel_weights)
        
        # 确保输出尺寸正确
        # 硬伤七修复: 使用 align_corners=False 保持像素位置对齐
        if out.shape[2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        
        # 4. Prediction
        main_logits = self.failure_head(out)
        main_prob = torch.sigmoid(main_logits)
        detail_input = torch.cat([
            out,
            f0,
            encoder_out['rgb0'],
            encoder_out['geo0'],
            encoder_out['geometry_maps'],
            main_prob,
        ], dim=1)
        detail_logits = self.detail_head(detail_input)
        final_logits = main_logits + self.detail_scale * detail_logits
        pred_failure = torch.sigmoid(final_logits)
        
        result = {
            'failure_logits': final_logits,
            'main_logits': main_logits,
            'detail_logits': detail_logits,
            'pred_failure': pred_failure,
            'pred_main': main_prob,
            'pred_detail': torch.sigmoid(detail_logits),
        }
        
        # 5. 深度监督 (可选)
        if self.deep_supervision and self.training:
            aux_failure_logits = self.aux_failure_heads(pyramid, (H, W))
            result['aux_failure_logits'] = aux_failure_logits
            result['aux_failure_maps'] = [torch.sigmoid(pred) for pred in aux_failure_logits]
        
        return result


# ============================================================================
# Factory Function
# ============================================================================

def create_bnd(
    vfm_type: str = 'moge2',
    vfm_checkpoint: Optional[str] = None,
    encoder_size: str = 'large',
    use_semantic_guidance: bool = True,
    deep_supervision: bool = True,
    **kwargs,
) -> nn.Module:
    """创建 V9 模型"""
    return BND(
        vfm_type=vfm_type,
        vfm_checkpoint=vfm_checkpoint,
        encoder_size=encoder_size,
        use_semantic_guidance=use_semantic_guidance,
        deep_supervision=deep_supervision,
    )


def _legacy_factory(
    version: str = 'v9',
    vfm_type: str = 'moge2',
    vfm_checkpoint: Optional[str] = None,
    encoder_size: str = 'large',
    use_semantic_guidance: bool = True,
    deep_supervision: bool = True,
    num_heads: int = 8,  # 兼容V4参数
    **kwargs,
) -> nn.Module:
    """
    统一工厂函数 - 兼容 V4/V5 调用接口
    
    Args:
        version: 'v4' / 'v5' / 'v9' (默认 v9)
        vfm_type: VFM 类型
        vfm_checkpoint: VFM 权重路径
        encoder_size: 编码器大小
        use_semantic_guidance: 是否使用语义引导 (V5 专属)
        deep_supervision: 是否使用深度监督
        num_heads: 注意力头数 (V4 兼容参数)
    """
    if version == 'v9':
        return BND(
            vfm_type=vfm_type,
            vfm_checkpoint=vfm_checkpoint,
            encoder_size=encoder_size,
            use_semantic_guidance=use_semantic_guidance,
            deep_supervision=deep_supervision,
        )
    elif version == 'v5':
        from .dual_stream_pixel_branch_v5 import DualStreamPixelBranchV5 as LegacyPixelBranchV5
        return LegacyPixelBranchV5(
            vfm_type=vfm_type,
            vfm_checkpoint=vfm_checkpoint,
            encoder_size=encoder_size,
            use_semantic_guidance=use_semantic_guidance,
            deep_supervision=deep_supervision,
        )
    elif version in ['v1', 'v2', 'v3', 'v4']:
        # 回退到 V4 实现
        from .dual_stream_pixel_branch_v4 import create_pixel_branch as create_v4
        return create_v4(
            version=version,
            vfm_type=vfm_type,
            vfm_checkpoint=vfm_checkpoint,
            encoder_size=encoder_size,
            num_heads=num_heads,
            deep_supervision=deep_supervision,
        )
    else:
        raise ValueError(f"Unknown version: {version}. Choose from 'v1', 'v2', 'v3', 'v4', 'v5', 'v9'")


# ============================================================================
# Backward Compatibility
# ============================================================================

DualStreamPixelBranch = BND


# ============================================================================
# 梯度检查点支持
# ============================================================================

def enable_gradient_checkpointing(model: BND):
    """为 V9 模型启用梯度检查点"""
    from torch.utils.checkpoint import checkpoint
    
    # 为 Encoder 启用
    original_encoder_forward = model.encoder.forward
    def checkpointed_encoder_forward(rgb, depth):
        return checkpoint(original_encoder_forward, rgb, depth, use_reentrant=False)
    model.encoder.forward = checkpointed_encoder_forward
    
    # 为 Decoder 启用
    original_decoder_forward = model.decoder.forward
    def checkpointed_decoder_forward(encoder_features, attention_maps=None):
        return checkpoint(original_decoder_forward, encoder_features, attention_maps, use_reentrant=False)
    model.decoder.forward = checkpointed_decoder_forward
    
    print("  ✓ Gradient checkpointing enabled for V9 model")
