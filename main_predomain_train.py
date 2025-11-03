import torch

from torch.optim import SGD

from model.CF_MAE import TemporalGeoMAE_ConvDecoder
from data.data_pretext import dataloader
from data.config import pre_text_samples

import time

import argparse


def argparses():
        parser = argparse.ArgumentParser()
        parser.add_argument("--pre_datafolder",type=str, required=True)
        parser.add_argument("--post_datafolder",type=str, required=True)

        return parser.parse_args()

       
def main(
        pre_datafolder,
        post_datafolder,
        epochs=100,
        img_size= 224,
        patch_size= 16,
        in_channels= 3,
        encoder_name= 'vit_base_patch16_224',
        encoder_embed_dim= 768,
        decoder_embed_dim = 512,
        mask_ratio= 0.75
        ):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'


        model = TemporalGeoMAE_ConvDecoder(
        
            img_size,
            patch_size,
            in_channels,
            encoder_name,
            encoder_embed_dim,
            decoder_embed_dim,
            mask_ratio)
        
        model.to(device)

        optimizer = SGD(model.parameters(), lr=0.01)


        train_loader = dataloader(
                
            pre_datafolder,
            post_datafolder,
            pre_text_samples.PRE_MEANSTD,
            pre_text_samples.POST_MEANSTD)
    
        print(F'[INITIALIZING THE TRAINING WITH-----{len(train_loader)*200}--NUMBER OF DATAS-----]')
        
        model.train()
        for i in range(1,epochs+1):
                
                start_time = time.time()
                t_loss = 0

                
                for images in train_loader:
                        
                        pre,post = images

                        pre.to(device)
                        post.to(device)

                        out = model(pre,post)

                        loss = model.compute_loss(
                                img_post = post,
                                pred_post = out['pred_post'],
                                mask_post = out['mask_post'],
                                change_logits = out['change_logits'])
                        
                        t_loss += loss['loss_recon'].item()

                        loss = loss['loss_recon']



                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()


                end_time = time.time()

                print(f'[EPOCHS-{i}]----[LOSS-{t_loss}]----[TIME-{start_time-end_time}]---------------------------------------')

        return None
                


if __name__ == "__main__":

    args = argparses()

    main(
        pre_datafolder = args.pre_datafolder,
        post_datafolder = args.post_datafolder
    )
    

                        

                        


        


