import yaml
from yaml.loader import FullLoader



def data_config(is_pretrain=True):
    with open("configs/data_config.yaml", "r") as f:
        data_config = yaml.load(f, Loader=FullLoader)

        if is_pretrain:
            return data_config['pretrain']
        else:
            return data_config['finetune']
        

def model_config(is_pretrain=True):
    with open("configs/model_config.yaml", "r") as f:
        model_config = yaml.load(f, Loader=FullLoader)

        if is_pretrain:
            return model_config['pretrain']
        else:
            return model_config['finetune']
        

def load_config(is_pretrain=True,type='data'):
    if type == 'data':
        return data_config(is_pretrain)
    elif type == 'model':
        return model_config(is_pretrain)
    else:
         raise NotImplementedError
