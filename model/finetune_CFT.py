import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class MS_CFTdv1(nn.Module):
    """
    A Convolutional decoder for Segmentation
    """
    def __init__(self,config):
        super().__init__()
        
        self.patch_size = config['ENCODER']['PATCH_SIZE']
        self.img_size = config['ENCODER']['IMG_SIZE']
        self.num_patches_per_dim = self.img_size // self.patch_size
        self.num_class = config['DECODER']['OUT_CHANNEL']
        self.embed_dim = config['DECODER']['EMBED_DIM']

        self.decoder = nn.Sequential(

            nn.Conv2d(self.embed_dim, 256*4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 128*4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 64*4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64*4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.num_class, kernel_size=1)
        )

        self.initialize_weights()
    
    def initialize_weights(self):

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
        
    def forward(self, x):
        
        B, N, D = x.shape
        H = W =  self.num_patches_per_dim
        x = x.transpose(2, 1).reshape(B, D, H, W)
        
        x = self.decoder(x) 
        x = self.final(x)

        return x
    
class MS_CFTev1(nn.Module):

    def __init__(self,config):
        super().__init__()

        self.config = config
        self.embed_dim = config['ENCODER']['EMBED_DIM']

        self.encoder = timm.create_model(
            config['ENCODER']['BACKBONE'],
            pretrained=False, 
            num_classes=0,
            drop_rate=config['ENCODER']['DROP_OUT'],
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

    def encoder4segm(self,x):

        x = self.encoder.patch_embed(x)

        if hasattr(self.encoder, 'pos_embed'):
            x = x + self.encoder.pos_embed[:, 1:, :] 

        for block in self.encoder.blocks:
            x = block(x)
        
        x = self.encoder.norm(x)

        return x

    def changeencode4Segm(self,pre,post):
        feature = torch.cat([pre,
                   post,
                   post-pre],dim=-1)
        
        z_diff = self.change_encoder(feature)
        return z_diff
    
    def forward(self,pre,post):
        pre = self.encoder4segm(pre)
        post = self.encoder4segm(post)

        z_pre = self.pre_projection(pre)
        z_post = self.post_projection(post)

        z_diff = self.changeencode4Segm(z_pre,z_post)
        x = self.decoder_embed(z_diff)

        return x


class MS_CFTdv2(nn.Module):
    """
    A Convolutional decoder for Segmentation with skip connections
    """
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config['DECODER']['EMBED_DIM']
        self.patch_size = config['ENCODER']['PATCH_SIZE']
        self.img_size = config['ENCODER']['IMG_SIZE']
        self.num_patches_per_dim = self.img_size // self.patch_size
        self.num_class = config['DECODER']['OUT_CHANNEL']

        self.up1 = nn.Sequential(
            nn.Conv2d(self.embed_dim, 256*4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(256 + self.embed_dim, 128*4, kernel_size=3, padding=1),  # +embed_dim for skip
            nn.PixelShuffle(2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.up3 = nn.Sequential(
            nn.Conv2d(128 + self.embed_dim, 64*4, kernel_size=3, padding=1),  # +embed_dim for skip
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.up4 = nn.Sequential(
            nn.Conv2d(64 + self.embed_dim, 64*4, kernel_size=3, padding=1),  # +embed_dim for skip
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.num_class, kernel_size=1)
        )
        self.initialize_weights()

    def initialize_weights(self):
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
        
    def forward(self, x, skip1=None, skip2=None, skip3=None):
        """
        Args:
            x: main feature from encoder (B, N, D)
            skip1, skip2, skip3: skip connections from intermediate layers (B, N, D)
        """
        B, N, D = x.shape
        H = W = self.num_patches_per_dim
        
        # Reshape main feature
        x = x.transpose(2, 1).reshape(B, D, H, W)
        
        # First upsampling (16x16 -> 32x32)
        x = self.up1(x)
        
        # Second upsampling with skip (32x32 -> 64x64)
        if skip3 is not None:
            skip3 = skip3.transpose(2, 1).reshape(B, D, H, W)
            skip3 = F.interpolate(skip3, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip3], dim=1)
        x = self.up2(x)
        
        # Third upsampling with skip (64x64 -> 128x128)
        if skip2 is not None:
            skip2 = skip2.transpose(2, 1).reshape(B, D, H, W)
            skip2 = F.interpolate(skip2, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip2], dim=1)
        x = self.up3(x)
        
        # Fourth upsampling with skip (128x128 -> 256x256)
        if skip1 is not None:
            skip1 = skip1.transpose(2, 1).reshape(B, D, H, W)
            skip1 = F.interpolate(skip1, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip1], dim=1)
        x = self.up4(x)
        
        x = self.final(x)

        return x


class MS_CFTev2(nn.Module):
    """
    Change Detection Segmentation model using pretrained CF-MAE encoder
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.img_size = config['ENCODER']['IMG_SIZE']
        self.patch_size = config['ENCODER']['PATCH_SIZE']
        self.encoder_name = config['ENCODER']['BACKBONE']
        self.embed_dim = config['ENCODER']['EMBED_DIM']

        self.encoder = timm.create_model(
            self.encoder_name,
            pretrained=False, 
            num_classes=0,
            drop_rate=config['ENCODER']['DROP_OUT'],
            drop_path_rate=config['ENCODER']['DROP_PATH'],
            global_pool='',
        )
        
        self.change_encoder1 = nn.Sequential(
            nn.Linear(self.embed_dim * 3, self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.Dropout(0.3)
        )

        self.change_encoder2 = nn.Sequential(
            nn.Linear(self.embed_dim * 3, self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.Dropout(0.3),
        )

        self.change_encoder3 = nn.Sequential(
            nn.Linear(self.embed_dim * 3, self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.Dropout(0.3))
        

        self.change_encoder4 = nn.Sequential(
            nn.Linear(self.embed_dim * 3, self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.Dropout(0.3)
        )

        # Decoder embed
        self.decoder_embed = nn.Sequential(nn.Linear(self.embed_dim ,self.embed_dim ),
                                           nn.GELU(),
                                           nn.Dropout(0.2),
                                            nn.Linear(self.embed_dim, self.embed_dim))
        
    
    def blocks1(self, x):

        x = self.encoder.patch_embed(x)

        if hasattr(self.encoder, 'pos_embed'):
            x = x + self.encoder.pos_embed[:, 1:, :]  


        for block in self.encoder.blocks[:3]:
            x = block(x)
        
        return x
    
    def blocks2(self, x):

        for block in self.encoder.blocks[3:6]:
            x = block(x)
        
        return x
    
    def blocks3(self, x):

        for block in self.encoder.blocks[6:9]:
            x = block(x)
        
        return x
    
    def blocks4(self, x):

        for block in self.encoder.blocks[9:]:
            x = block(x)
        
        x = self.encoder.norm(x)

        return x

    def changeencode(self,change_module, pre, post):

        feature = torch.cat([
            pre,
            post,
            post - pre
        ], dim=-1)
        
        z_diff = change_module(feature)
        return z_diff
    
    def forward(self, pre, post):

        pre1 = self.blocks1(pre)
        post1 = self.blocks1(post)
        z_diff1 = self.changeencode(self.change_encoder1,pre1,post1)

        pre2 = self.blocks2(pre1)
        post2 = self.blocks2(post1)
        z_diff2 = self.changeencode(self.change_encoder2,pre2,post2)

        pre3 = self.blocks3(pre2)
        post3 = self.blocks3(post2)
        z_diff3 = self.changeencode(self.change_encoder3,pre3,post3)

        pre4 = self.blocks4(pre3)
        post4 = self.blocks4(post3)
        z_diff4 = self.changeencode(self.change_encoder4,pre4,post4)

        z_dec = self.decoder_embed(z_diff4)
        
        return z_dec,z_diff1,z_diff2,z_diff3
    

class MS_CFTev3(nn.Module):
    """
    Change Detection Segmentation model using pretrained CF-MAE encoder
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.img_size = config['ENCODER']['IMG_SIZE']
        self.patch_size = config['ENCODER']['PATCH_SIZE']
        self.encoder_name = config['ENCODER']['BACKBONE']
        self.embed_dim = config['ENCODER']['EMBED_DIM']

        self.encoder = timm.create_model(
            self.encoder_name,
            pretrained=False, 
            num_classes=0,
            drop_rate=config['ENCODER']['DROP_OUT'],
            drop_path_rate=config['ENCODER']['DROP_PATH'],
            global_pool='',
        )
        
        self.change_encoder1 = nn.Sequential(
            nn.Linear(self.embed_dim * 3, self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.Dropout(0.3)
        )

        self.change_encoder2 = nn.Sequential(
            nn.Linear(self.embed_dim * 3, self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.Dropout(0.3),
        )

        self.change_encoder3 = nn.Sequential(
            nn.Linear(self.embed_dim * 3, self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.Dropout(0.2))
        

        self.change_encoder4 = nn.Sequential(
            nn.Linear(self.embed_dim * 3, self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.Dropout(0.2)
        )

        # Decoder embed
        self.decoder_embed = nn.Sequential(nn.Linear(self.embed_dim*4 ,self.embed_dim*2 ),
                                           nn.GELU(),
                                           nn.Dropout(0),
                                            nn.Linear(self.embed_dim*2, self.embed_dim))
    
    def blocks1(self, x):

        x = self.encoder.patch_embed(x)

        if hasattr(self.encoder, 'pos_embed'):
            x = x + self.encoder.pos_embed[:, 1:, :]  


        for block in self.encoder.blocks[:3]:
            x = block(x)
        
        return x
    
    def blocks2(self, x):

        for block in self.encoder.blocks[3:6]:
            x = block(x)
        
        return x
    
    def blocks3(self, x):

        for block in self.encoder.blocks[6:9]:
            x = block(x)
        
        return x
    
    def blocks4(self, x):

        for block in self.encoder.blocks[9:]:
            x = block(x)
        
        x = self.encoder.norm(x)

        return x

    def changeencode(self,change_module, pre, post):

        feature = torch.cat([
            pre,
            post,
            post - pre
        ], dim=-1)
        
        z_diff = change_module(feature)
        return z_diff
    
    def MS_feature(self,diff1,diff2,diff3,diff4):

        feature = torch.cat([
            diff1,
            diff2,
            diff3,
            diff4
        ], dim=-1)
        
        z_f = self.decoder_embed(feature)
        return z_f

    
    def forward(self, pre, post):

        pre1 = self.blocks1(pre)
        post1 = self.blocks1(post)
        z_diff1 = self.changeencode(self.change_encoder1,pre1,post1)

        pre2 = self.blocks2(pre1)
        post2 = self.blocks2(post1)
        z_diff2 = self.changeencode(self.change_encoder2,pre2,post2)

        pre3 = self.blocks3(pre2)
        post3 = self.blocks3(post2)
        z_diff3 = self.changeencode(self.change_encoder3,pre3,post3)

        pre4 = self.blocks4(pre3)
        post4 = self.blocks4(post3)
        z_diff4 = self.changeencode(self.change_encoder4,pre4,post4)

        z_dec = self.MS_feature(z_diff1,z_diff2,z_diff3,z_diff4)
        
        return z_dec


class MS_CFT4Segm(nn.Module):
    """
    Change Detection Segmentation model using pretrained CF-MAE encoder
    """
    def __init__(self, config):
        super().__init__()

        self.version = config["VERSION"]

        print(f"[INFO]--Building Version {self.version} for segmentation")

        if self.version == "v1":
            self.encoder  = MS_CFTev1(config)
            self.decoder = MS_CFTdv1(config)
        
        elif self.version == "v2":
            self.encoder  = MS_CFTev2(config)
            self.decoder = MS_CFTdv2(config)

        elif self.version == "v3":
            self.encoder = MS_CFTev3(config)
            self.decoder = MS_CFTdv1(config)

        else:
            raise ValueError(
                f"Unknown VERSION {self.version!r}; expected one of v1, v2, v3"
            )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'encoder.encoder.pos_embed'}
    
    def forward(self, pre, post):

        if self.version == "v1":
            z_dec = self.encoder(pre,post)
            pred = self.decoder(z_dec)
        elif self.version == "v2":
            z_dec,skip1,skip2,skip3 = self.encoder(pre,post)
            pred = self.decoder(z_dec,skip1,skip2,skip3)
        elif self.version == "v3":
            z_dec = self.encoder(pre,post)
            pred = self.decoder(z_dec)

        return pred
    
    def compute_loss(self, pred, label):
        criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
        return criterion(pred, label)
    
class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: (B, 1, H, W) or (B, H, W)
        targets: same shape, values in {0,1}
        """
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice.mean()
    
class BCEDiceLoss(nn.Module):

    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

        

 