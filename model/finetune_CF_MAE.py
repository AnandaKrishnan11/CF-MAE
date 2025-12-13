import torch
import torch.nn as nn
import torch.nn.functional as F
from CF_MAE import CF_MAE

class Decoder4Segm(nn.Module):
    """
    A Convolutional decoder for Segmentation
    """
    def __init__(
        self,
        config,
        embed_dim,
        img_size,
        patch_size,
        num_class
    ):
        super().__init__()
        
        self.patch_size = config['ENCODER']['PATCH_SIZE']
        self.img_size = config['ENCODER']['IMG_SIZE']
        self.num_patches_per_dim = self.img_size // self.patch_size
        self.num_class = config['DECODER']['OUT_CHANNEL']

        self.decoder = nn.Sequential(
        
                nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.num_class, kernel_size=1)
        )
        
    def forward(self, x):
        
        B, N, D = x.shape
        H = W =  self.num_patches_per_dim
        x = x.transpose(2, 1).reshape(B, D, H, W)
        
        x = self.decoder(x) 
        x = F.interpolate(x,size=(224,224),mode='bilinear', align_corners=False)
        x = self.final(x)

        return x
    
class CF_MAE4Segm(nn.Module):

    def __init__(self,config):
        super().__init__()

        self.config = config

        self.model = CF_MAE(config)
        try:
            checkpoint = torch.load(self.config['PRE_TRAINED_CHKPT'])
            self.model.load_state_dict(checkpoint["model"])
        except Exception as e:
            print(e)

        self.decoder = Decoder4Segm(config)
        self.decoder._is_decoder = True

        self.initialize_weights()
    
    def initialize_weights(self):

        # Initialize mask token
        nn.init.normal_(self.mask_token, std=0.02)
        # Initialize linear layers
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if not hasattr(m, "_is_decoder"):
            return  # skiping pretrained weights init 
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
        return {'encoder.cls_token', 'encoder.pos_embed'}
    
    def compute_loss(self,pred,label):
        criterion = nn.BCEWithLogitsLoss()
        return criterion(pred,label)

    def encoder4segm(self,x):

        x = self.model.encoder.patch_embed(x)

        if hasattr(self.model, 'pos_embed'):
            x = x + self.model.encoder.pos_embed[:, 1:, :] 

        for block in self.model.encoder.blocks:
            x = block(x)
        
        x = self.model.encoder.norm(x)

        return x

    def changeencode4Segm(self,pre,post):
        feature = torch.cat([pre,
                   post,
                   post-pre],dim=-1)
        
        z_diff = self.model.change_encoder(feature)
        return z_diff
    
    def forward(self,pre,post):
        pre = self.encoder4segm(pre)
        post = self.encoder4segm(post)

        z_pre = self.model.pre_projection(pre)
        z_post = self.model.post_projection(post)

        z_diff = self.changeencode4Segm(z_pre,z_post)
        x = self.decoder(z_diff)

        return x



         
         

