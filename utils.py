import torch
import matplotlib.pyplot as plt
import os

def load_pretrained_cfmae(model, config):
    """
    Load CF-MAE pretrained weights and initialize multi-scale change encoders
    for MS-CFTv1 AND v2
    """
    checkpoint_path = config["TRAINING"]["PRETRAINED_WEIGHT"]
    print(f"Loading CF-MAE pretrained weights from: {checkpoint_path} for {config['VERSION']}")

    if config["VERSION"] == "v1":

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"]

        encoder_state_dict = {}
        change_encoder_base_state = {}
        decoder_embed_state_dict = {}
        pre_proj = {}
        post_proj = {}

        for k,v in state_dict.items():
            if k.startswith("mask_token") or k.startswith("decoder."):
                    continue
            elif k.startswith("encoder."):
                new_key = k.replace("encoder.","")
                encoder_state_dict[new_key] = v
            elif k.startswith("change_encoder."):
                new_key = k.replace("change_encoder.","")
                change_encoder_base_state[new_key] = v
            elif k.startswith("decoder_embed."):
                new_key = k.replace("decoder_embed.", "")
                decoder_embed_state_dict[new_key] = v
            elif k.startswith("pre_projection."):
                new_key = k.replace("pre_projection.", "")
                pre_proj[new_key] = v
            elif k.startswith("post_projection."):
                new_key = k.replace("post_projection.", "")
                post_proj[new_key] = v
            else:
                encoder_state_dict[k] = v

        model.encoderv1.encoder.load_state_dict(encoder_state_dict)
        model.encoderv1.pre_projection.load_state_dict(pre_proj)
        model.encoderv1.post_projection.load_state_dict(post_proj)
        model.encoderv1.change_encoder.load_state_dict(change_encoder_base_state)
        model.encoderv1.decoder_embed.load_state_dict(decoder_embed_state_dict)

        return model,True
    else:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint["model"]
            
            # Containers
            encoder_state_dict = {}
            change_encoder_base_state = {}
            decoder_embed_state_dict = {}
            
            # Parse checkpoint
            for k, v in state_dict.items():
                if k.startswith("mask_token") or k.startswith("decoder."):
                    continue
                elif k.startswith("change_encoder."):
                    # Extract base change encoder weights
                    new_key = k.replace("change_encoder.", "")
                    change_encoder_base_state[new_key] = v
                elif k.startswith("encoder."):
                    new_key = k.replace("encoder.", "")
                    encoder_state_dict[new_key] = v
                elif k.startswith("decoder_embed."):
                    new_key = k.replace("decoder_embed.", "")
                    decoder_embed_state_dict[new_key] = v
                else:
                    encoder_state_dict[k] = v
            
            # Load encoder weights
            missing, unexpected = model.encoderv2.encoder.load_state_dict(
                encoder_state_dict, strict=False
            )
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
            print("✓ Loaded encoder weights")
            
            # Initialize all change encoders from pretrained weights
            if change_encoder_base_state:
                # Map pretrained 3-layer to new 5-layer architecture
                # Pretrained: [Linear(0), GELU(1), Linear(2)]
                # New:        [Linear(0), GELU(1), Dropout(2), Linear(3), Dropout(4)]
                
                def init_change_encoder(change_encoder_module):
                    """Initialize a 5-layer change encoder from 3-layer pretrained weights"""
                    new_state = {}
                    
                    for key, value in change_encoder_base_state.items():
                        if key.startswith('0.'):  # First Linear layer
                            new_state[key] = value
                        elif key.startswith('2.'):  # Second Linear layer (was index 2, now index 3)
                            new_key = key.replace('2.', '3.')
                            new_state[new_key] = value
                    
                    # Load the mapped weights (strict=False to allow Dropout layers to be randomly initialized)
                    change_encoder_module.load_state_dict(new_state, strict=False)
                    print(f"  ✓ Initialized change encoder with pretrained Linear layers")
                
                # Initialize all 4 change encoders
                init_change_encoder(model.encoderv2.change_encoder1)
                init_change_encoder(model.encoderv2.change_encoder2)
                init_change_encoder(model.encoderv2.change_encoder3)
                init_change_encoder(model.encoderv2.change_encoder4)
                
                print("✓ Initialized all change_encoder modules (1-4) from CF-MAE")

            if config["VERSION"] == "v2":
            
                # Load decoder_embed
                if decoder_embed_state_dict:
                    # Pretrained decoder_embed is a single Linear layer
                    # New decoder_embed is a 4-layer Sequential: [Linear, GELU, Dropout, Linear]
                    
                    # Map the single Linear layer to the first Linear in the Sequential
                    new_decoder_state = {}
                    for key, value in decoder_embed_state_dict.items():
                        # Map to the first layer (index 0) of the Sequential
                        new_key = f"0.{key}"
                        new_decoder_state[new_key] = value
                    
                    # The last Linear layer (index 3) will use the same weights as initialization
                    for key, value in decoder_embed_state_dict.items():
                        new_key = f"3.{key}"
                        new_decoder_state[new_key] = value
                    
                    missing_dec, unexpected_dec = model.encoderv2.decoder_embed.load_state_dict(
                        new_decoder_state, strict=False
                    )
                    print("✓ Loaded decoder_embed weights (first and last Linear layers)")
            
            print("\n✓ Successfully loaded and initialized all weights")
            return model, True
            
        except Exception as e:
            print(f"[Error] Failed to load pretrained weights: {e}")
            import traceback
            traceback.print_exc()
            print("⚠ Continuing with model initialization")
            return model, False

class image:
    def __init__(self):
        self.id = 0

    def visualise_scene(self,pre_img,post_img,label_img,pred_img):

        batch_count = pre_img.shape[0]

        for idx in range(batch_count):

            pre = pre_img[idx,:,:,:].cpu().numpy().transpose(1,2,0)
            post = post_img[idx,:,:,:].cpu().numpy().transpose(1,2,0)
            label = label_img[idx,:,:,:].cpu().numpy().transpose(1,2,0)
            pred = pred_img[idx,:,:,:].cpu().numpy().transpose(1,2,0)

            fig,axs = plt.subplots(1,4,figsize = (15,5))

            axs[0].imshow(pre)
            axs[0].set_title("Pre Image")
            axs[0].axis("off")

            axs[1].imshow(post)
            axs[1].set_title("Post Image")
            axs[1].axis("off")

            axs[2].imshow(label,cmap = "gray")
            axs[2].set_title("GT")
            axs[2].axis("off")
            axs[3].imshow(pred, cmap ="gray")
            axs[3].set_title("MS-CFT")
            axs[3].axis("off")

            plt.tight_layout()
            out_dir = '/media/krishnan/GFM/SEGMENTATION IMAGES/MY_MODELS/DSIFN/MS-CFT'
            os.makedirs(out_dir,exist_ok=True)
            plt.savefig(f'{out_dir}/img{self.id}.png',dpi=400)
            self.id+=1
            plt.close()
