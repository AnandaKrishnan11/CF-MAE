import yaml
from yaml.loader import FullLoader



def data_config(is_pretrain=True,data=None):
    with open("configs/data_config.yaml", "r") as d:
        data_config = yaml.load(d, Loader=FullLoader)

        if is_pretrain:
            return data_config['PRETRAIN']
        else:
            if data == 'S2WCD':
                return data_config['FINETUNE']['S2WCD']
            elif data == 'DSIFN':
                return data_config['FINETUNE']['DSIFN']
            elif data == 'MAUI':
                return data_config['FINETUNE']['MAUI']
        

def model_config(is_pretrain=True):
    with open("configs/model_config.yaml", "r") as m:
        model_config = yaml.load(m, Loader=FullLoader)

        if is_pretrain:
            return model_config['PRETRAIN']
        else:
            return model_config['FINETUNE']
        

def load_config(data=None,is_pretrain=True,type='data'):
    if type == 'data':
        return data_config(is_pretrain,data)
    elif type == 'model':
        return model_config(is_pretrain)
    else:
         raise NotImplementedError
    

def write_config(config,type='data'):
    if type == 'data':
        with open("configs/data_config.yaml", "w") as d:
            yaml.safe_dump(config, d, sort_keys=False)
    elif type == 'model':
        with open("configs/model_config.yaml", "w") as m:
            yaml.safe_dump(config, m, sort_keys=False)
    else:
         raise NotImplementedError