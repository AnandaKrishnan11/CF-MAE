import torch

from model.build_model import build_model
from data.finetune_loader import build_loader 
from configs.build_config import load_config
from logger import create_logger
from metrics import metric_calculation

import os
import argparse

def parse_args():
      parser = argparse.ArgumentParser('CF-MAE inference')
      parser.add_argument('--data-path', default=None, help='override the dataset root')
      parser.add_argument('--output', default=None, help='override the log directory')
      parser.add_argument('--checkpoint', default=None, help='path to the checkpoint to evaluate')
      parser.add_argument('--version', default=None, choices=['v1','v2','v3'], help='model version')
      parser.add_argument('--batch-size', type=int, default=None)
      parser.add_argument('--run', default='test', choices=['train','val','test'], help='split to evaluate')
      return parser.parse_args()


def apply_args(model_config, data_config, args):
      if args.data_path is not None:
            data_config['PATH'] = args.data_path
      if args.batch_size is not None:
            data_config['BATCH_SIZE'] = args.batch_size
      if args.version is not None:
            model_config['VERSION'] = args.version
      if args.checkpoint is not None:
            model_config['TRAINING']['BEST_CHECKPOINT_PATH'] = args.checkpoint
      if args.output is not None:
            model_config['TRAINING']['LOGGER_PATH'] = args.output
      return model_config, data_config

def main():

      args = parse_args()

      data_config = load_config(is_pretrain=False,type='data')
      model_config = load_config(is_pretrain=False,type='model')
      model_config, data_config = apply_args(model_config, data_config, args)

      device = 'cuda' if torch.cuda.is_available() else 'cpu'

      output_dir = model_config['TRAINING']['LOGGER_PATH']
      os.makedirs(output_dir, exist_ok=True)
      logger = create_logger(output_dir, dist_rank=0, name='inference_logger')
      logger.info(f'Data Config - {data_config}')
      logger.info(f'Model Config - {model_config}')

      logger.info(f"Creating model:{model_config['NAME']} version {model_config['VERSION']}")
      model = build_model(model_config,is_pretrain=False)

      checkpoint_path = model_config['TRAINING']['BEST_CHECKPOINT_PATH']
      logger.info(f'Loading checkpoint {checkpoint_path}')
      checkpoint = torch.load(checkpoint_path, map_location="cpu")
      model.load_state_dict(checkpoint['model'], strict=True)
      logger.info(f"Checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

      model = model.to(device)
      logger.info(str(model))

      n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
      print(f"[INFO]----Model initialized with {n_parameters} number of paramter")
      logger.info(f"number of params: {n_parameters}")

      test_loader = build_loader(data_config,logger,run=args.run)
      metric = metric_calculation()

      model.eval()
      with torch.no_grad():
            for pre,post,lbl in test_loader:
                  pre = pre.to(device)
                  post = post.to(device)
                  lbl = lbl.to(device)

                  pred = model(pre,post)
                  pred_bin = (torch.sigmoid(pred) > 0.5).long()

                  metric.compute_metrics(lbl,pred_bin)

      vals = metric.get_metrics()

      logger.info(f"[SPLIT]----------------{args.run} ({len(test_loader.dataset)} samples)")
      logger.info(f"[IOU METRICS]----------------{vals['iou']}")
      logger.info(f"[F1 METRICS]----------------{vals['f1']}")

      print(f"[RESULT]----{args.run}: IoU {vals['iou']:.4f} | F1 {vals['f1']:.4f}")


if __name__ =='__main__':
      main()