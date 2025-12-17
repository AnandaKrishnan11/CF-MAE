import torch

from model.build_model import build_model
from data.finetune_loader import build_loader 
from configs.build_config import load_config
from logger import create_logger
from metrics import metric_calculation

def main():
      
      data_config = load_config(is_pretrain=False,type='data')
      model_config = load_config(is_pretrain=False,type='model')

      output_dir = data_config['PATH']
      logger = create_logger(output_dir, dist_rank=0, name='pretrain_logger')

      logger.info(f"Creating model:{model_config['NAME']}")
      model = build_model(model_config,name = model_config['NAME'])
      try:
        checkpoint = torch.load(model_config['FINETUNE_WEIGHT_S2WCD'], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
      except Exception as e:
        print(f'Error-------------{e}')

      model.cuda()
      logger.info(str(model))

      n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
      print(f"[INFO]----Model initialized with {n_parameters} number of paramter")
      logger.info(f"number of params: {n_parameters}")

      val_loader = build_loader(data_config,logger,train=False)

      metric = metric_calculation()

      with torch.no_grad():
            model.eval()

            for pre,post,lbl in val_loader:
                  pre = pre.cuda(non_blocking=True)
                  post = post.cuda(non_blocking=True)
                  lbl = lbl.cuda(non_blocking=True)

                  pred = model(pre,post)
                  pred = torch.sigmoid(pred)
                  pred_bin = (pred > 0.5).long()

                  metric.compute_metrics(lbl,pred_bin)

            vals = metric.get_metrics()

            print(f"[IOU METRICS]----------------{vals['iou']}")
            print(f"[F1 METRICS]----------------{vals['f1']}")
            print(f"[PRECISION METRICS]----------------{vals['precision']}")
            print(f"[RECALL METRICS]----------------{vals['recall']}")

if __name__ =='__main__':
    main()
