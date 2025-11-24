import torch
from timm.utils import AverageMeter

from model.build_model import build_model
from data.pretext_loader import build_loader 
from configs.build_config import load_config,write_config
from logger import create_logger
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from utils import get_grad_norm

import os
import time
from tqdm import tqdm
import datetime

def main():
      
      #Building config for model and data
      data_config = load_config(is_pretrain=True,type='data')
      model_config = load_config(is_pretrain=True,type='model')

      device = 'cpu'

      output_dir = data_config['PATH']
      logger = create_logger(output_dir, dist_rank=0, name='pretrain_logger')
      
      logger.info(f"Creating model:{model_config['MODEL']['TYPE']} style--{model_config['MODEL']['NAME']}")
      print(f"[INFO]---Creating model:{model_config['MODEL']['TYPE']} style--{model_config['MODEL']['NAME']}")
      model = build_model(model_config,logger,name = model_config['MODEL']['NAME'])
      model.to(device)
      logger.info(str(model))

      optimizer = build_optimizer(model_config,model,logger,is_pretrain=True)

      n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
      print(f"[INFO]----Model initialized with {n_parameters} number of paramter")
      logger.info(f"number of params: {n_parameters}")

      train_loader = build_loader(data_config,logger)

      print(f"[INFO]----Total number of Images used {data_config['TOTAL_SAMPLES']}")
      print(f"[INFO]----Image size {data_config['IMAGE_SIZE']}")
      print(f"[INFO]----Image size {data_config['NUM_CHANNELS']}")
      print(f"[INFO]----Batch size used {data_config['BATCH_SIZE']}")

      lr_scheduler = build_scheduler(model_config,logger, optimizer,n_iter_per_epoch=len(train_loader))

      #Resume model training
      if model_config['TRAINING']['AUTO_RESUME'] and os.path.exists(model_config['TRAINING']['CHECKPOINT_PATH']):
            print(f"[INFO]----Resuming training from checkpoint: {model_config['TRAINING']['CHECKPOINT_PATH']}")
            checkpoint = torch.load(model_config['TRAINING']['CHECKPOINT_PATH'], map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            model_config['TRAINING']['START_EPOCH'] = checkpoint["epoch"] + 1
            print(f"[INFO]----Restarting training at epoch: {model_config['TRAINING']['START_EPOCH']}")

      #train
      start_epoch = model_config['TRAINING']['START_EPOCH']
      total_epochs  = model_config['TRAINING']['TOTAL_EPOCHS']
      num_steps = len(train_loader)

      batch_time = AverageMeter()
      loss_meter = AverageMeter()
      norm_meter = AverageMeter()

      model.train()
      for epoch in tqdm(range(start_epoch,total_epochs),total=total_epochs):
            e_start = time.time()

            for idx,(pre,post) in enumerate(train_loader,start=1):  
                  b_start = time.time()
                  pre=pre.to(device)
                  post=post.to(device)

                  loss = model(pre,post)
                  
                  #Gradient Accumulation
                  if model_config['TRAINING']['ACCUMULATION_STEPS']>1:
                        loss = loss['loss_recon'] / model_config['TRAINING']['ACCUMULATION_STEPS']
                        loss.backward()
                        if model_config['TRAINING']['GRAD_CLIP']:
                              grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), model_config['TRAINING']['GRAD_CLIP'])       
                        else:
                              grad_norm = get_grad_norm(model.parameters())  
                        if idx % model_config['TRAINING']['ACCUMULATION_STEPS'] == 0  or (idx == num_steps):
                              optimizer.step()
                              optimizer.zero_grad()
                              lr_scheduler.step_update(epoch * num_steps + idx)
                  else:
                        optimizer.zero_grad()        
                        loss.backward()
                        if model_config['TRAINING']['GRAD_CLIP']:
                              grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), model_config['TRAINING']['GRAD_CLIP'])       
                        else:
                              grad_norm = get_grad_norm(model.parameters())  
                  
                        optimizer.step()
                        lr_scheduler.step_update(epoch * num_steps + idx)

                  torch.cuda.synchronize()

                  loss_meter.update(loss.item(),n=pre.size(0))
                  norm_meter.update(grad_norm)
                  batch_time.update(time.time()-b_start)

                  if idx % model_config['TRAINING']['PRINT_FREQ'] == 0:
                        lr = optimizer.param_groups[0]['lr']
                        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                        etas = batch_time.avg * (num_steps - idx)
                        logger.info(
                        f"Train: [{epoch}/{model_config['TRAINING']['TOTAL_EPOCHS']}][{idx}/{num_steps}]\t"
                        f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                        f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                        f'mem {memory_used:.0f}MB')
            
            save_state = {'model': model.state_dict(),'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(),'epoch': epoch}
            torch.save(save_state,os.path.join(model_config['TRAINING']['CHECKPOINT_PATH'],f'ckpt_epoch_{epoch}.pth'))
            model_config['TRAINING']['CHECKPOINT_PATH'] = os.path.join(model_config['TRAINING']['CHECKPOINT_PATH'],f'ckpt_epoch_{epoch}.pth')
            write_config(config=model_config,type="model")
            logger.info(f"{model_config['TRAINING']['CHECKPOINT_PATH']} saved !!!")

            epoch_time = time.time() - e_start
            logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
            print(f"[INFO][EPOCH]---{epoch}|[TIME]---{datetime.timedelta(seconds=int(epoch_time))}|[LOSS]---{loss_meter.avg:.4f}")
                      
if __name__ == "__main__":
      main()
    

                        

                        


        


