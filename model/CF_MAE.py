#-----------------------------------------------
# References:
# mae : https://github.com/facebookresearch/mae.git
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
#-----------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional


class ConvDecoder(nn.Module):
    """
    Convolutional decoder for better spatial reconstruction
    Progressively upsamples from patch tokens to full image
    """
    def __init__(
        self,
        decoder_embed_dim: int = 512,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches_per_dim = img_size // patch_size  # 14 for 224/16
        
        # Initial spatial dimension after reshaping tokens
        # For 224x224 with patch_size=16: 14x14 grid
        init_size = self.num_patches_per_dim
        
        # Project to higher channel dimension for convolutions
        self.input_proj = nn.Conv2d(decoder_embed_dim, 512, 1)
        
        # Progressive upsampling with residual blocks
        # Path: 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
        self.upsample_blocks = nn.ModuleList([
            # 14x14 -> 28x28
            UpsampleBlock(512, 256, scale_factor=2),
            # 28x28 -> 56x56
            UpsampleBlock(256, 128, scale_factor=2),
            # 56x56 -> 112x112
            UpsampleBlock(128, 64, scale_factor=2),
            # 112x112 -> 224x224
            UpsampleBlock(64, 32, scale_factor=2),
        ])
        
        # Final prediction head
        self.output_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, 1),
        )
        
    def forward(self, x, ids_restore, mask_token):
        """
        Args:
            x: (B, N_visible, D) - visible patch embeddings
            ids_restore: (B, N) - indices to restore full sequence
            mask_token: (1, 1, D) - learnable mask token
        Returns:
            pred: (B, C, H, W) - reconstructed image
        """
        B, N_visible, D = x.shape
        N = self.num_patches_per_dim ** 2
        
        # Expand mask token for all masked positions
        mask_tokens = mask_token.repeat(B, N - N_visible, 1)
        
        # Concatenate visible and mask tokens
        x_full = torch.cat([x, mask_tokens], dim=1)  # (B, N, D)
        
        # Restore original order using ids_restore
        x_full = torch.gather(
            x_full, 
            dim=1, 
            index=ids_restore.unsqueeze(-1).repeat(1, 1, D)
        )  # (B, N, D)
        
        # Reshape to spatial grid: (B, D, H, W)
        H = W = self.num_patches_per_dim
        x_spatial = x_full.transpose(1, 2).reshape(B, D, H, W)
        
        # Project to conv channels
        x = self.input_proj(x_spatial)  # (B, 512, 14, 14)
        
        # Progressive upsampling
        for upsample_block in self.upsample_blocks:
            x = upsample_block(x)
        
        # Final prediction
        pred = self.output_conv(x)  # (B, C, 224, 224)
        
        return pred


class UpsampleBlock(nn.Module):
    """
    Upsampling block with residual connection
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        
        self.upsample = nn.Upsample(
            scale_factor=scale_factor, 
            mode='bilinear', 
            align_corners=False
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # Residual connection
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )
        
    def forward(self, x):
        # Upsample
        x_up = self.upsample(x)
        
        # Main path
        out = self.conv1(x_up)
        out = self.conv2(out)
        
        # Residual path
        residual = self.residual(x_up)
        
        return F.relu(out + residual)


class TemporalGeoMAE_ConvDecoder(nn.Module):
    """
    Temporal MAE with CONVOLUTIONAL decoder for better reconstruction
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        encoder_name: str = 'vit_base_patch16_224',
        encoder_embed_dim: int = 768,
        decoder_embed_dim: int = 512,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        
        self.num_patches = (img_size // patch_size) ** 2
        self.encoder_embed_dim = encoder_embed_dim
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # Shared encoder
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=False,
            num_classes=0,
            global_pool='',
        )
        
        # Projections for latent alignment
        self.pre_projection = nn.Sequential(
            nn.Linear(encoder_embed_dim, encoder_embed_dim),
            nn.GELU(),
            nn.Linear(encoder_embed_dim, encoder_embed_dim),
        )
        
        self.post_projection = nn.Sequential(
            nn.Linear(encoder_embed_dim, encoder_embed_dim),
            nn.GELU(),
            nn.Linear(encoder_embed_dim, encoder_embed_dim),
        )
        
        # Change encoder (rich difference)
        self.change_encoder = nn.Sequential(
            nn.Linear(encoder_embed_dim * 4, encoder_embed_dim * 2),
            nn.GELU(),
            nn.Linear(encoder_embed_dim * 2, encoder_embed_dim),
        )
        
        # Decoder embedding (projects concatenated features)
        self.decoder_embed = nn.Linear(
            encoder_embed_dim * 2,  # z_pre + z_diff
            decoder_embed_dim
        )
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # CONVOLUTIONAL DECODER
        self.decoder = ConvDecoder(
            decoder_embed_dim=decoder_embed_dim,
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
        )
        
        # Change detection head
        self.change_classifier = nn.Sequential(
            nn.Linear(encoder_embed_dim, encoder_embed_dim // 2),
            nn.GELU(),
            nn.Linear(encoder_embed_dim // 2, 2),
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize mask token
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Initialize linear layers
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def random_masking(self, x, mask_ratio):
        """
        Perform random masking by per-sample shuffling.
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def encode(self, img, mask_ratio):
        """Encode image with masking"""
        # Get patch embeddings from encoder
        x = self.encoder.patch_embed(img)
        
        # Add position embeddings
        if hasattr(self.encoder, 'pos_embed'):
            x = x + self.encoder.pos_embed[:, 1:, :]  # Skip cls token

        # Random masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Apply encoder blocks
        for block in self.encoder.blocks:
            x = block(x)
        
        x = self.encoder.norm(x)
        
        return x, mask, ids_restore
    
    def compute_change_encoding(self, z_pre, z_post):
        """Compute rich difference encoding"""
        # Concatenate multiple change representations
        z_concat = torch.cat([
            z_pre,
            z_post,
            z_post - z_pre,  # Difference
            z_post * z_pre,  # Interaction
        ], dim=-1)
        
        z_diff = self.change_encoder(z_concat)
        return z_diff
    
    def forward(self, img_pre, img_post):
        # Encode both images
        z_pre_enc, mask_pre, ids_restore_pre = self.encode(img_pre, self.mask_ratio)
        z_post_enc, mask_post, ids_restore_post = self.encode(img_post, self.mask_ratio)
        
        # Project to aligned space
        z_pre = self.pre_projection(z_pre_enc)
        z_post = self.post_projection(z_post_enc)
        
        # Compute change encoding
        z_diff = self.compute_change_encoding(z_pre, z_post)
        
        # Concatenate z_pre and z_diff for decoder
        z_combined = torch.cat([z_pre, z_diff], dim=-1)
        z_dec = self.decoder_embed(z_combined)
        
        # CONVOLUTIONAL DECODE (returns full image, not patches)
        pred_post = self.decoder(z_dec, ids_restore_post, self.mask_token)
        
        # Change detection
        z_diff_pooled = z_diff.mean(dim=1)
        change_logits = self.change_classifier(z_diff_pooled)
        
        return {
            'pred_post': pred_post,  # (B, C, H, W)
            'mask_post': mask_post,
            'change_logits': change_logits,
            'z_pre': z_pre,
            'z_post': z_post,
            'z_diff': z_diff,
        }
    
    def patchify(self, imgs):
        """
        imgs: (B, C, H, W)
        x: (B, N, patch_size**2 * C)
        """
        p = self.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p**2 * 3)
        return x
    
    def compute_loss(
        self,
        img_post: torch.Tensor,
        pred_post: torch.Tensor,
        mask_post: torch.Tensor,
        change_logits: torch.Tensor
    ):
        """
        Compute loss for convolutional decoder
        Uses pixel-wise MSE instead of patch-wise
        """
        B, C, H, W = img_post.shape
        
        # Reshape mask to spatial dimensions
        mask_spatial = mask_post.reshape(B, H // self.patch_size, W // self.patch_size)
        mask_spatial = mask_spatial.unsqueeze(1).repeat(1, C, 1, 1)  # (B, C, h, w)
        
        # Upsample mask to full resolution
        mask_full = F.interpolate(
            mask_spatial.float(), 
            size=(H, W), 
            mode='nearest'
        )
        
        # Pixel-wise MSE loss on masked regions
        loss_recon = (pred_post - img_post) ** 2
        loss_recon = (loss_recon * mask_full).sum() / (mask_full.sum() + 1e-6)
        
        loss_dict = {'loss_recon': loss_recon}
    
        return loss_dict


# Example usage
if __name__ == "__main__":
    model = TemporalGeoMAE_ConvDecoder(
        img_size=224,
        patch_size=16,
        in_channels=3,
        encoder_embed_dim=768,
        decoder_embed_dim=512,
        mask_ratio=0.75,
    )
    
    # Dummy input
    img_pre = torch.randn(2, 3, 224, 224)
    img_post = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    outputs = model(img_pre, img_post)
    
    print("Prediction shape:", outputs['pred_post'].shape)  # (2, 3, 224, 224)
    print("Mask shape:", outputs['mask_post'].shape)  # (2, 196)
    print("Change logits shape:", outputs['change_logits'].shape)  # (2, 2)
    
    # Compute loss
    loss_dict = model.compute_loss(
        img_post=img_post,
        pred_post=outputs['pred_post'],
        mask_post=outputs['mask_post'],
        change_logits=outputs['change_logits'],
        change_labels=torch.randint(0, 2, (2,)),
    )
    
    print("Loss:", loss_dict['loss_total'].item())