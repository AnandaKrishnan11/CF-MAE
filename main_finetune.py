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
import argparse
import datetime
from tqdm import tqdm


def parse_args():
      parser = argparse.ArgumentParser('CF-MAE finetuning')
      parser.add_argument('--data-path', default=None, required= True, help='override the dataset root')
      parser.add_argument('--output', default=None, required= True, help='override the log/checkpoint directory')
      parser.add_argument('--version', default="v2", choices=['v1','v2','v3'], help='model version')
      parser.add_argument('--pretrained', required= True, default=None, help='path to the CF-MAE checkpoint')
      parser.add_argument('--batch-size', type=int, default=16)
      parser.add_argument('--lr', type=float, default=5e-5)
      parser.add_argument('--epochs', type=int, default=50)
      parser.add_argument('--accumulation-steps', type=int, default=0)
      parser.add_argument('--no-resume', action='store_true', help='ignore the last checkpoint')
      return parser.parse_args()


def apply_args(model_config, data_config, args):
      if args.data_path is not None:
            data_config['PATH'] = args.data_path
      if args.batch_size is not None:
            data_config['BATCH_SIZE'] = args.batch_size
      if args.version is not None:
            model_config['VERSION'] = args.version
      if args.lr is not None:
            model_config['TRAINING']['LEARNING_RATE'] = args.lr
      if args.epochs is not None:
            model_config['TRAINING']['TOTAL_EPOCHS'] = args.epochs
      if args.accumulation_steps is not None:
            model_config['TRAINING']['ACCUMULATION_STEPS'] = args.accumulation_steps
      if args.pretrained is not None:
            model_config['TRAINING']['PRETRAINED_WEIGHT'] = args.pretrained
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
      data_config = load_config(is_pretrain=False,type='data')
      model_config = load_config(is_pretrain=False,type='model')
      model_config, data_config = apply_args(model_config, data_config, args)

      device = 'cuda' if torch.cuda.is_available() else 'cpu'

      output_dir = model_config['TRAINING']['LOGGER_PATH']
      os.makedirs(output_dir, exist_ok=True)
      logger = create_logger(output_dir, dist_rank=0, name='finetune_logger')

      #Snapshot of the exact settings used for this run
      write_config({'FINETUNE': model_config}, output_dir, 'model_config.yaml')
      write_config({'FINETUNE': data_config}, output_dir, 'data_config.yaml')

      logger.info(f'Data Config - {data_config}')
      logger.info(f'Model Config - {model_config}')
      
      logger.info(f"Creating model:{model_config['TYPE']} style--{model_config['NAME']}")
      print(f"[INFO]---Creating model:{model_config['TYPE']} style--{model_config['NAME']}")
      model = build_model(model_config,is_pretrain=False)

      model = load_pretrained_cfmae(model,model_config)
      model.to(device)
      logger.info(str(model))

      optimizer = build_optimizer(model_config,model,logger,is_pretrain=False)

      n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
      print(f"[INFO]----Model initialized with {n_parameters} number of paramter")
      logger.info(f"number of params: {n_parameters}")

      train_loader = build_loader(data_config,logger,run='train')
      val_loader = build_loader(data_config,logger,run='val')

      logger.info(f"[INFO]----Total number of Images used {len(train_loader.dataset)}")
      logger.info(f"[INFO]----Image size {data_config['IMAGE_SIZE']}")
      logger.info(f"[INFO]----Image channels {data_config['NUM_CHANNELS']}")
      logger.info(f"[INFO]----Batch size used {data_config['BATCH_SIZE']}")

      lr_scheduler = build_scheduler(model_config,logger, optimizer,n_iter_per_epoch=len(train_loader))

      best_iou = -1.0 #We are only saving the best model and latest one
      early_stop_counter = 0

      #Resume model training
      if model_config['TRAINING']['AUTO_RESUME'] and os.path.isfile(model_config['TRAINING']['LAST_CHECKPOINT_PATH']):
            print(f"[INFO]----Resuming training from checkpoint: {model_config['TRAINING']['LAST_CHECKPOINT_PATH']}")
            checkpoint = torch.load(model_config['TRAINING']['LAST_CHECKPOINT_PATH'], map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            model_config['TRAINING']['START_EPOCH'] = checkpoint["epoch"] + 1
            best_iou = checkpoint.get("best_iou", -1.0)
            early_stop_counter = checkpoint.get("early_stop_counter", 0)
            print(f"[INFO]----Restarting training at epoch: {model_config['TRAINING']['START_EPOCH']}")

      #train
      start_epoch = model_config['TRAINING']['START_EPOCH']
      total_epochs  = model_config['TRAINING']['TOTAL_EPOCHS']
      num_steps = len(train_loader)
      grad_clip = model_config['TRAINING']['GRAD_CLIP']
      accumulation_steps = int(model_config['TRAINING']['ACCUMULATION_STEPS'])

      for epoch in tqdm(range(start_epoch,total_epochs),total=total_epochs-start_epoch):
            e_start = time.time()
            batch_time = AverageMeter()
            loss_meter = AverageMeter()
            val_loss_meter = AverageMeter()

            model.train()
            optimizer.zero_grad()
            for idx,(pre,post,lbl) in enumerate(train_loader,start=1):  
                  b_start = time.time()
                  pre = pre.to(device)
                  post = post.to(device)
                  lbl = lbl.to(device)

                  pred = model(pre,post)
                  loss = model.compute_loss(pred,lbl)
                  loss_value = loss.item()

                  #Gradient Accumulation
                  if accumulation_steps > 1:
                        loss = loss / accumulation_steps
                        loss.backward()
                        if grad_clip:
                              # as I understood, the gradient is cliped in place, I would rather not save the grad norms  
                              # as indicated here https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html#torch.nn.utils.clip_grad_norm_
                              torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                        if idx % accumulation_steps == 0 or (idx == num_steps):
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

                  loss_meter.update(loss_value,n=pre.size(0))
                  batch_time.update(time.time()-b_start)

                  if idx % model_config['TRAINING']['PRINT_FREQ'] == 0:
                        lr = optimizer.param_groups[0]['lr']
                        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                        etas = batch_time.avg * (num_steps - idx)
                        logger.info(
                        f"Train: [{epoch}/{total_epochs}][{idx}/{num_steps}]\t"
                        f'eta {datetime.timedelta(seconds=int(etas))}\t'
                        f'Lr: {lr} \t'
                        f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'mem {memory_used:.0f}MB')

            metric = metric_calculation()
            model.eval()
            with torch.no_grad():
                  for idx,(pre,post,lbl) in enumerate(val_loader,start=1):
                        pre = pre.to(device)
                        post = post.to(device)
                        lbl = lbl.to(device)

                        pred = model(pre,post)
                        val_loss = model.compute_loss(pred,lbl)
                        val_loss_meter.update(val_loss.item(),n=pre.size(0))

                        pred_bin = (torch.sigmoid(pred) > 0.5).long()
                        metric.compute_metrics(lbl,pred_bin)

            vals = metric.get_metrics()

            logger.info(f"[IOU METRICS]----------------{vals['iou']}")
            logger.info(f"[F1 METRICS]----------------{vals['f1']}")

            epoch_time = time.time() - e_start
            logger.info(f"[INFO][EPOCH]---{epoch}|[TIME]---{datetime.timedelta(seconds=int(epoch_time))}|[TRAIN-LOSS]---{loss_meter.avg:.4f}|[VAL-LOSS]---{val_loss_meter.avg:.4f}")

            save_state = {'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'lr_scheduler': lr_scheduler.state_dict(),
                          'epoch': epoch,
                          'best_iou': best_iou,
                          'early_stop_counter': early_stop_counter}

            if vals['iou'] > best_iou:
                  best_iou = vals['iou']
                  early_stop_counter = 0
                  save_state['best_iou'] = best_iou
                  save_state['early_stop_counter'] = early_stop_counter
                  torch.save(save_state,model_config['TRAINING']['BEST_CHECKPOINT_PATH'])
                  logger.info(f"[BEST CHECK POINT]--{model_config['TRAINING']['BEST_CHECKPOINT_PATH']} saved !!! iou {best_iou:.4f}")
            else:
                  early_stop_counter += 1
                  save_state['early_stop_counter'] = early_stop_counter
                  logger.info(f'No improvement in the last {early_stop_counter} epochs')

            torch.save(save_state,model_config['TRAINING']['LAST_CHECKPOINT_PATH'])
            logger.info(f"[LAST CHECK POINT]--{model_config['TRAINING']['LAST_CHECKPOINT_PATH']} saved !!!")

            if early_stop_counter >= model_config['TRAINING']['PATIENCE']:
                  logger.info(f'[INIT]---------EARLYSTOPING AT EPOCH--{epoch}')
                  break


if __name__ == "__main__":
      main()