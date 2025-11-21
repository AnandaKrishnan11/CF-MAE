import yaml
from yaml.loader import FullLoader



def data_config(is_pretrain=True):
    with open("configs/data_config.yaml", "r") as d:
        data_config = yaml.load(d, Loader=FullLoader)

        if is_pretrain:
            return data_config['PRETRAIN']
        else:
            return data_config['FINETUNE']
        

def model_config(is_pretrain=True):
    with open("configs/model_config.yaml", "r") as m:
        model_config = yaml.load(m, Loader=FullLoader)

        if is_pretrain:
            return model_config['PRETRAIN']
        else:
            return model_config['FINETUNE']
        

def load_config(is_pretrain=True,type='data'):
    if type == 'data':
        return data_config(is_pretrain)
    elif type == 'model':
        return model_config(is_pretrain)
    else:
         raise NotImplementedError
