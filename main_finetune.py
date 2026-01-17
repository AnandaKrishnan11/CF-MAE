import torch
from timm.utils import AverageMeter

from model.build_model import build_model
from data.finetune_loader import build_loader 
from configs.build_config import load_config,write_config
from logger import create_logger
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from utils import load_pretrained_cfmae
from metrics import metric_calculation

import os
import time
from tqdm import tqdm
import datetime

def main():
      
      #Building config for model and data
      data_config = load_config(is_pretrain=False,type='data',data='DSIFN')
      model_config = load_config(is_pretrain=False,type='model')

      device = 'cuda' if torch.cuda.is_available() else 'cpu'

      output_dir = model_config['TRAINING']['LOGGER_PATH']
      logger = create_logger(output_dir, dist_rank=0, name='SATMAEtrail2_finetune_logger')

      logger.info(f'Data Config - {data_config}')
      logger.info(f'Model Config - {model_config}')
      
      logger.info(f"Creating model:{model_config['TYPE']} style--{model_config['NAME']}")
      print(f"[INFO]---Creating model:{model_config['TYPE']} style--{model_config['NAME']}")
      model = build_model(model_config,is_pretrain=False)

      if model_config['TRAINING']['USE_PRETRAINED_WEIGHT']:
        model,success = load_pretrained_cfmae(model,model_config)
        if not success:
            logger.info("Failed to load pretrained weights, using random initialization")
      model.to(device)
      logger.info(str(model))

      optimizer = build_optimizer(model_config,model,logger,is_pretrain=False)

      n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
      print(f"[INFO]----Model initialized with {n_parameters} number of paramter")
      logger.info(f"number of params: {n_parameters}")

      train_loader = build_loader(data_config,logger,run='train',data='DSIFN')
      val_loader = build_loader(data_config,logger,run='val',data='DSIFN')

      logger.info(f"[INFO]----Total number of Images used {len(train_loader.dataset)}")
      logger.info(f"[INFO]----Image size {data_config['IMAGE_SIZE']}")
      logger.info(f"[INFO]----Image channels {data_config['NUM_CHANNELS']}")
      logger.info(f"[INFO]----Batch size used {data_config['BATCH_SIZE']}")

      lr_scheduler = build_scheduler(model_config,logger, optimizer,n_iter_per_epoch=len(train_loader))

      #Resume model training
      if model_config['TRAINING']['AUTO_RESUME'] and os.path.isfile(model_config['TRAINING']['LAST_CHECKPOINT_PATH']):
            print(f"[INFO]----Resuming training from checkpoint: {model_config['TRAINING']['LAST_CHECKPOINT_PATH']}")
            checkpoint = torch.load(model_config['TRAINING']['LAST_CHECKPOINT_PATH'], map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            model_config['TRAINING']['START_EPOCH'] = checkpoint["epoch"] + 1
            write_config(model_config,type='model')
            print(f"[INFO]----Restarting training at epoch: {model_config['TRAINING']['START_EPOCH']}")

      #train
      start_epoch = model_config['TRAINING']['START_EPOCH']
      total_epochs  = model_config['TRAINING']['TOTAL_EPOCHS']
      num_steps = len(train_loader)
      min_loss = 0 #We are only saving the best model and latest one

      early_stop_counter = 0

      for epoch in tqdm(range(start_epoch,total_epochs),total=total_epochs):
            e_start = time.time()
            batch_time = AverageMeter()
            loss_meter = AverageMeter()
            val_loss_meter = AverageMeter()

            model.train()
            for idx,(pre,post,lbl) in enumerate(train_loader,start=1):  
                  b_start = time.time()
                  pre = pre.to(device)
                  post = post.to(device)
                  lbl = lbl.to(device)

                  pred = model(pre,post)
                  loss = model.compute_loss(pred,lbl)
                  
                  #Gradient Accumulation
                  if model_config['TRAINING']['ACCUMULATION_STEPS']>1:
                        loss = loss / model_config['TRAINING']['ACCUMULATION_STEPS']
                        loss.backward()
                        if model_config['TRAINING']['GRAD_CLIP']: 
                              # as I understood, the gradient is cliped in place, I would rather not save the grad norms  
                              # as indicated here https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html#torch.nn.utils.clip_grad_norm_
                              torch.nn.utils.clip_grad_norm_(model.parameters(), model_config['TRAINING']['GRAD_CLIP'])       

                        if idx % model_config['TRAINING']['ACCUMULATION_STEPS'] == 0  or (idx == num_steps):
                              optimizer.step()
                              optimizer.zero_grad()
                              lr_scheduler.step_update(epoch * num_steps + idx)
                  else:
                        optimizer.zero_grad()        
                        loss.backward()
                        if model_config['TRAINING']['GRAD_CLIP']:
                              torch.nn.utils.clip_grad_norm_(model.parameters(), model_config['TRAINING']['GRAD_CLIP'])      
                  
                        optimizer.step()
                        lr_scheduler.step_update(epoch * num_steps + idx)

                  loss_meter.update(loss.item(),n=pre.size(0))
                  batch_time.update(time.time()-b_start)

                  if idx % model_config['TRAINING']['PRINT_FREQ'] == 0:
                        lr = optimizer.param_groups[0]['lr']
                        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                        etas = batch_time.avg * (num_steps - idx)
                        logger.info(
                        f"Train: [{epoch}/{model_config['TRAINING']['TOTAL_EPOCHS']}][{idx}/{num_steps}]\t"
                        f'eta {datetime.timedelta(seconds=int(etas))}\t'
                        f'Lr: {lr} \t'
                        f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'mem {memory_used:.0f}MB')

            metric = metric_calculation()
            with torch.no_grad():
                  model.eval()
                  for idx,(pre,post,lbl) in enumerate(val_loader,start=1):
                        pre = pre.to(device)
                        post = post.to(device)
                        lbl = lbl.to(device)

                        pred = model(pre,post)

                        pred = torch.sigmoid(pred)
                        pred_bin = (pred > 0.5).long()
                        metric.compute_metrics(lbl,pred_bin)

            vals = metric.get_metrics()

            logger.info(f"[IOU METRICS]----------------{vals['iou']}")
            logger.info(f"[F1 METRICS]----------------{vals['f1']}")
            logger.info(f"[PRECISION METRICS]----------------{vals['precision']}")
            logger.info(f"[RECALL METRICS]----------------{vals['recall']}")
                        #val_loss = model.compute_loss(pred,lbl)
                        #val_loss_meter.update(val_loss.item(),n=pre.size(0))

            epoch_time = time.time() - e_start
            logger.info(f"[INFO][EPOCH]---{epoch}|[TIME]---{datetime.timedelta(seconds=int(epoch_time))}|[TRAIN-LOSS]---{loss_meter.avg:.4f}|[VAL-LOSS]---{val_loss_meter.avg:.4f}")
            
            if min_loss==0 or min_loss < vals['iou']: #val_loss_meter.avg
                  min_loss = vals['iou'] #val_loss_meter.avg
                  save_state = {'model': model.state_dict(),'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(),'epoch': epoch}
                  torch.save(save_state,model_config['TRAINING']['BEST_CHECKPOINT_PATH'])
                  logger.info(f"[BEST CHECK POINT]--{model_config['TRAINING']['BEST_CHECKPOINT_PATH']} saved !!!")
                  torch.save(save_state,model_config['TRAINING']['LAST_CHECKPOINT_PATH'])
                  logger.info(f"[LAST CHECK POINT]--{model_config['TRAINING']['LAST_CHECKPOINT_PATH']} saved !!!")
                  early_stop_counter = 0
            else:
                  torch.save(save_state,model_config['TRAINING']['LAST_CHECKPOINT_PATH'])
                  logger.info(f"[LAST CHECK POINT]--{model_config['TRAINING']['LAST_CHECKPOINT_PATH']} saved !!!")
                  early_stop_counter +=1
                  logger.info(f'No improvement in the last {early_stop_counter} epochs')
                  
                  if early_stop_counter >= model_config['TRAINING']['PATIENCE']:
                        logger.info(f'[INIT]---------EARLYSTOPING AT EPOCH--{epoch}')
                        break


                      
if __name__ == "__main__":
      main()
    

                        

                        


        


