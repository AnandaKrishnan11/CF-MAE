#-----------------------------------------------
# References:
# mae : https://github.com/facebookresearch/mae.git
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# SimMIM: https://github.com/microsoft/SimMIM
#-----------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class simDecoder(nn.Module):
    """
    A simple Convolutional decoder for spatial reconstruction
    """
    def __init__(
        self,
        embed_dim,
        img_size,
        patch_size
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches_per_dim = img_size // patch_size
        

        self.decoder = nn.Sequential(nn.Conv2d(
            in_channels=embed_dim,
            out_channels=patch_size ** 2 * 3, 
            kernel_size=1),
            nn.PixelShuffle(patch_size))
        
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
        
        mask_tokens = mask_token.repeat(B, N - N_visible, 1)
        
        x_full = torch.cat([x, mask_tokens], dim=1)  
        
        x_full = torch.gather(
            x_full, 
            dim=1, 
            index=ids_restore.unsqueeze(-1).repeat(1, 1, D)
        ) 
        
        H = W = self.num_patches_per_dim
        x_spatial = x_full.transpose(2,1).reshape(B, D, H, W)
        
        pred = self.decoder(x_spatial) 
        return pred

class CF_MAE(nn.Module):
    """
    Temporal MAE with CONVOLUTIONAL decoder 
    """
    def __init__(
        self,
        config
    ):
        super().__init__()

        self.img_size = config['ENCODER']['IMG_SIZE']
        self.patch_size = config['ENCODER']['PATCH_SIZE']
        self.encoder_name = config['ENCODER']['BACKBONE']
        self.mask_ratio = config['MASKING']['MASK_RATIO']
        self.embed_dim = config['ENCODER']['EMBED_DIM']
        
        self.encoder = timm.create_model(
            self.encoder_name,
            pretrained=True,
            num_classes=0,
            drop_path_rate=config['ENCODER']['DROP_PATH'],
            global_pool='',
        ) 

        self.pre_projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        
        self.post_projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.change_encoder = nn.Sequential(
            nn.Linear(self.embed_dim * 3, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
        )
        
        self.decoder_embed = nn.Linear(
            self.embed_dim,
            self.embed_dim
        )
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        self.decoder = simDecoder(
            embed_dim=self.embed_dim,
            img_size=self.img_size,
            patch_size=self.patch_size
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)

        # only the from-scratch modules; the timm encoder keeps its pretrained weights
        for module in (self.pre_projection, self.post_projection, self.change_encoder,
                    self.decoder_embed, self.decoder):
            module.apply(self._init_weights)

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
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'encoder.cls_token', 'encoder.pos_embed','mask_token'}
    
    def random_masking(self, x, mask_ratio, noise=None):
        """
        Perform random masking by per-sample shuffling.
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        if noise is None:
            noise = torch.rand(B, N, device=x.device)
        
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore, noise
    
    def encode(self, img, mask_ratio,noise):
        """Encode image with masking"""

        x = self.encoder.patch_embed(img)
        
        if hasattr(self.encoder, 'pos_embed'):
            x = x + self.encoder.pos_embed[:, 1:, :]  # Skip cls token

        x, mask, ids_restore,_ = self.random_masking(x, mask_ratio,noise)
        
        for block in self.encoder.blocks:
            x = block(x)
        
        x = self.encoder.norm(x)
        
        return x, mask, ids_restore
    
    def compute_change_encoding(self, z_pre, z_post):
        """Compute rich difference encoding"""

        z_concat = torch.cat([
            z_pre,
            z_post,
            z_post - z_pre
        ], dim=-1)
        
        z_diff = self.change_encoder(z_concat)
        return z_diff
    
    def forward(self, img_pre, img_post): 

        B = img_pre.shape[0]
        N = (self.img_size // self.patch_size) ** 2
        noise = torch.rand(B, N, device=img_pre.device)
     
        z_pre_enc, mask_pre, ids_restore_pre = self.encode(img_pre, self.mask_ratio,noise)
        z_post_enc, mask_post, ids_restore_post = self.encode(img_post, self.mask_ratio,noise)
        
        z_pre = self.pre_projection(z_pre_enc)
        z_post = self.post_projection(z_post_enc)
        
        z_diff = self.compute_change_encoding(z_pre, z_post)
        
        z_dec = self.decoder_embed(z_diff)
        
        pred_post = self.decoder(z_dec, ids_restore_post, self.mask_token)
        
        #Computing loss
        loss = self.compute_loss(img_post,pred_post,mask_post)
        
        return loss
    
    def compute_loss(
        self,
        img_post,
        pred_post,
        mask_post,
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
        loss_recon = (loss_recon * mask_full).sum() / (mask_full.sum() + 1e-5)

        return loss_recon
