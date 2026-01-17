from model.pretrain_CF_MAE import CF_MAE
from model.finetune_CF_MAE import MS_CFT4Segm


def build_model(config,is_pretrain = True):

    if is_pretrain:
        return CF_MAE(config)
    else:
        return MS_CFT4Segm(config)
 


