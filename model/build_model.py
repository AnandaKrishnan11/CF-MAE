from model.pretrain_CF_MAE import CF_MAE
from model.finetune_CF_MAE import CF_MAE4Segm


def build_model(config,is_pretrain = True):

    if is_pretrain:
        return CF_MAE(config)
    else:
        return CF_MAE4Segm(config)
 


