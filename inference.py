import torch

from model.build_model import build_model
from data.finetune_loader import build_loader 
from configs.build_config import load_config
from logger import create_logger
from metrics import metric_calculation
from utils import image

def main():
      
      data_config = load_config(is_pretrain=False,type='data',data="S2WCD")
      model_config = load_config(is_pretrain=False,type='model')

      device = 'cuda' if torch.cuda.is_available() else 'cpu'

      output_dir = data_config['PATH']
      logger = create_logger(output_dir, dist_rank=0, name='inference_logger')
      logger.info(f'Data Config - {data_config}')
      logger.info(f'Model Config - {model_config}')

      logger.info(f"Creating model:{model_config['NAME']}")
      model = build_model(model_config,is_pretrain=False)

      
      checkpoint = torch.load(model_config['TRAINING']['BEST_CHECKPOINT_PATH'], map_location="cpu")
      #new_check = {}
      #for k,v in checkpoint["model"].items():
          #if k.startswith('encoder4segm'):
               #new = k.replace('encoder4segm','encoderv2')
               #new_check[new] = v

          #elif k.startswith('decoder4segm'):
               #new = k.replace('decoder4segm','decoderv2')
               #new_check[new] = 

      missing,unexpected = model.load_state_dict(checkpoint['model'],strict=True)
      print(f'[MISSING]---{missing}')
      print(f'[UNEXPECTED]---{unexpected}')

      model = model.to(device)
      logger.info(str(model))

      n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
      print(f"[INFO]----Model initialized with {n_parameters} number of paramter")
      logger.info(f"number of params: {n_parameters}")

      test_loader = build_loader(data_config,logger,run ='test',data="S2WCD")
      img = image()
      metric = metric_calculation()

      with torch.no_grad():
            model.eval()
            for pre,post,lbl in test_loader:
                  pre = pre.to(device)
                  post = post.to(device)
                  lbl = lbl.to(device)

                  pred = model(pre,post)
                  pred = torch.sigmoid(pred)
                  pred_bin = (pred > 0.5).long()

                  img.visualise_scene(pre,post,lbl,pred_bin)

                  metric.compute_metrics(lbl,pred_bin)
                  

            vals = metric.get_metrics()

            logger.info(f"[IOU METRICS]----------------{vals['iou']}")
            logger.info(f"[F1 METRICS]----------------{vals['f1']}")
            logger.info(f"[PRECISION METRICS]----------------{vals['precision']}")
            logger.info(f"[RECALL METRICS]----------------{vals['recall']}")

if __name__ =='__main__':
    main()