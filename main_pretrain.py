import torch
from timm.utils import AverageMeter

from model.build_model import build_model
from data.pretext_loader import build_loader 
from configs.build_config import load_config,write_config
from logger import create_logger
from lr_scheduler import build_scheduler
from optimizer import build_optimizer

import os
import time
import argparse
import datetime
from tqdm import tqdm


def parse_args():
      parser = argparse.ArgumentParser('CF-MAE pretraining')
      parser.add_argument('--data-path', default=None, required= True, help='override the dataset root')
      parser.add_argument('--output', default=None, required= True, help='override the log/checkpoint directory')
      parser.add_argument('--batch-size', type=int, default=128)
      parser.add_argument('--lr', type=float, default=0.0001)
      parser.add_argument('--epochs', type=int, default=100)
      parser.add_argument('--accumulation-steps', type=int, default=0)
      parser.add_argument('--no-resume', action='store_true', help='ignore the last checkpoint')
      return parser.parse_args()


def apply_args(model_config, data_config, args):
      if args.data_path is not None:
            data_config['PATH'] = args.data_path
      if args.batch_size is not None:
            data_config['BATCH_SIZE'] = args.batch_size
      if args.lr is not None:
            model_config['TRAINING']['LEARNING_RATE'] = args.lr
      if args.epochs is not None:
            model_config['TRAINING']['TOTAL_EPOCHS'] = args.epochs
      if args.accumulation_steps is not None:
            model_config['TRAINING']['ACCUMULATION_STEPS'] = args.accumulation_steps
      if args.no_resume:
            model_config['TRAINING']['AUTO_RESUME'] = False
      if args.output is not None:
            model_config['TRAINING']['LOGGER_PATH'] = args.output
            model_config['TRAINING']['BEST_CHECKPOINT_PATH'] = os.path.join(args.output, 'best_chk_pt.pth')
            model_config['TRAINING']['LAST_CHECKPOINT_PATH'] = os.path.join(args.output, 'last_chk_pt.pth')
      return model_config, data_config


def main():

      args = parse_args()

      #Building config for model and data
      data_config = load_config(is_pretrain=True,type='data')
      model_config = load_config(is_pretrain=True,type='model')
      model_config, data_config = apply_args(model_config, data_config, args)

      output_dir = model_config["TRAINING"]['LOGGER_PATH']
      os.makedirs(output_dir, exist_ok=True)
      logger = create_logger(output_dir, dist_rank=0, name='pretrain_logger')

      #Snapshot of the exact settings used for this run
      write_config({'PRETRAIN': model_config}, output_dir, 'model_config.yaml')
      write_config({'PRETRAIN': data_config}, output_dir, 'data_config.yaml')

      logger.info(f'Data Config - {data_config}')
      logger.info(f'Model Config - {model_config}')
      
      logger.info(f"Creating model:{model_config['TYPE']} style--{model_config['NAME']}")
      print(f"[INFO]---Creating model:{model_config['TYPE']} style--{model_config['NAME']}")
      model = build_model(model_config,is_pretrain=True)
      model.cuda()
      logger.info(str(model))

      optimizer = build_optimizer(model_config,model,logger)

      n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
      print(f"[INFO]----Model initialized with {n_parameters} number of paramter")
      logger.info(f"number of params: {n_parameters}")

      train_loader = build_loader(data_config,logger)

      print(f"[INFO]----Total number of Images used {len(train_loader.dataset)}")
      print(f"[INFO]----Image size {data_config['IMAGE_SIZE']}")
      print(f"[INFO]----Number of channels {data_config['NUM_CHANNELS']}")
      print(f"[INFO]----Batch size used {data_config['BATCH_SIZE']}")

      lr_scheduler = build_scheduler(model_config,logger, optimizer,n_iter_per_epoch=len(train_loader))

      min_loss = float('inf') #We are only saving the best model

      #Resume model training
      if model_config['TRAINING']['AUTO_RESUME'] and os.path.isfile(model_config['TRAINING']['LAST_CHECKPOINT_PATH']):
            print(f"[INFO]----Resuming training from checkpoint: {model_config['TRAINING']['LAST_CHECKPOINT_PATH']}")
            checkpoint = torch.load(model_config['TRAINING']['LAST_CHECKPOINT_PATH'], map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            model_config['TRAINING']['START_EPOCH'] = checkpoint["epoch"] + 1
            min_loss = checkpoint.get("min_loss", float('inf'))
            print(f"[INFO]----Restarting training at epoch: {model_config['TRAINING']['START_EPOCH']}")

      #train
      start_epoch = model_config['TRAINING']['START_EPOCH']
      total_epochs  = model_config['TRAINING']['TOTAL_EPOCHS']
      num_steps = len(train_loader)
      grad_clip = model_config['TRAINING']['GRAD_CLIP']
      accumulation_steps = int(model_config['TRAINING']['ACCUMULATION_STEPS'])

      model.train()
      optimizer.zero_grad()
      for epoch in tqdm(range(start_epoch,total_epochs),total=total_epochs-start_epoch):
            loss_meter = AverageMeter()
            batch_time = AverageMeter()
            e_start = time.time()

            for idx,(pre,post) in enumerate(train_loader,start=1):  
                  b_start = time.time()
                  pre = pre.cuda(non_blocking=True)
                  post = post.cuda(non_blocking=True)

                  loss = model(pre,post)
                  loss_value = loss.item()

                  #Gradient Accumulation
                  if accumulation_steps > 1:
                        loss = loss / accumulation_steps
                        loss.backward()
                        if grad_clip:
                              torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                        if idx % accumulation_steps == 0 or idx == num_steps:
                              optimizer.step()
                              optimizer.zero_grad()
                              lr_scheduler.step_update(epoch * num_steps + idx)
                  else:
                        optimizer.zero_grad()
                        loss.backward()
                        if grad_clip:
                              torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()
                        lr_scheduler.step_update(epoch * num_steps + idx)

                  torch.cuda.synchronize()

                  loss_meter.update(loss_value,n=pre.size(0))
                  batch_time.update(time.time()-b_start)

                  if idx % model_config['TRAINING']['PRINT_FREQ'] == 0:
                        lr = optimizer.param_groups[0]['lr']
                        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                        etas = batch_time.avg * (num_steps - idx)
                        logger.info(
                        f"Train: [{epoch}/{total_epochs}][{idx}/{num_steps}]\t"
                        f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                        f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'mem {memory_used:.0f}MB')

            save_state = {'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'lr_scheduler': lr_scheduler.state_dict(),
                          'epoch': epoch,
                          'min_loss': min_loss}

            torch.save(save_state,model_config['TRAINING']['LAST_CHECKPOINT_PATH'])
            logger.info(f"[LAST CHECK POINT]--{model_config['TRAINING']['LAST_CHECKPOINT_PATH']} saved !!!")

            if loss_meter.avg < min_loss:
                  min_loss = loss_meter.avg
                  save_state['min_loss'] = min_loss
                  torch.save(save_state,model_config['TRAINING']['BEST_CHECKPOINT_PATH'])
                  logger.info(f"[BEST CHECK POINT]--{model_config['TRAINING']['BEST_CHECKPOINT_PATH']} saved !!! loss {min_loss:.4f}")

            epoch_time = time.time() - e_start
            logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
            print(f"[INFO][EPOCH]---{epoch}|[TIME]---{datetime.timedelta(seconds=int(epoch_time))}|[LOSS]---{loss_meter.avg:.4f}")


if __name__ == "__main__":
      main()