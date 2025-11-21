import torch

from model.build_model import build_model
from data.pretext_loader import build_loader 
from configs.build_config import load_config
from logger import create_logger
from lr_scheduler import build_scheduler
from optimizer import build_optimizer

import time
import argparse


def argparses():
        parser = argparse.ArgumentParser()
        parser.add_argument("--pre_datafolder",type=str, required=True)
        parser.add_argument("--post_datafolder",type=str, required=True)

        return parser.parse_args()

       
def main(config):
      
      #Building config for model and data
      data_config = load_config(is_pretrain=True,type='data')
      model_config = load_config(is_pretrain=True,type='model')

      logger = create_logger(data_config, dist_rank=0, name='pretrain_logger')
      
      train_loader = build_loader(data_config,logger)

      logger.info(f"Creating model:{model_config['MODEL']['TYPE']} style--{model_config['MODEL']['NAME']}")
      model = build_model(model_config,logger,name = model_config['MODEL']['NAME'])
      model.cuda()
      logger.info(str(model))

      optimizer = build_optimizer(model_config,model,logger,is_pretrain=True)

      n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
      print(f"[INFO]----Model initialized with {n_parameters} number of paramter")
      logger.info(f"number of params: {n_parameters}")

      lr_scheduler = build_scheduler(config,logger, optimizer,n_iter_per_epoch=len(train_loader))

      #Resume model training
      if model_config['TRAINING']['AUTO_RESUME'] and model_config['TRAINING']['CHECKPOINT_PATH'].exists():
            print(f"Resuming training from checkpoint: {model_config['TRAINING']['CHECKPOINT_PATH']}")
            checkpoint = torch.load(model_config['TRAINING']['CHECKPOINT_PATH'], map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            model_config['TRAINING']['START_EPOCH'] = checkpoint["epoch"] + 1
            print(f"Resuming training at epoch: {model_config['TRAINING']['START_EPOCH']}")

      #train
      start_epoch = model_config['TRAINING']['START_EPOCH']
      total_epochs  = model_config['TRAINING']['TOTAL_EPOCHS']
      num_steps = len(train_loader)

      model.train()
      for epochs in range(start_epoch,total_epochs):

            for idx,(pre,post) in enumerate(train_loader):  
                  optimizer.zero_grad()        
                  pre.cuda()
                  post.cuda()

                  loss = model(pre,post)# integrate loss function to the model
                  
                  loss.backward()
                  optimizer.step()
                  lr_scheduler.step_update(epochs * num_steps + idx)

            torch.cuda.synchronize()


                  

                  



      #epoch  learning rate  loss  time



      

            
            



if __name__ == "__main__":

    args = argparses()

    main()
    

                        

                        


        


