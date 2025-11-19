# --------------------------------------------------------
# Based from SimMIM codebase
# https://github.com/microsoft/SimMIM
# --------------------------------------------------------

from torch import optim as optim


def build_optimizer(config, model, logger, is_pretrain):
    if is_pretrain:
        return build_pretrain_optimizer(config, model, logger)
    else:
        raise NotImplementedError


def build_pretrain_optimizer(config, model, logger):

    logger.info('>>>>>>>>>> Build Optimizer for Pre-training Stage')
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
        logger.info(f'No weight decay: {skip}')
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
        logger.info(f'No weight decay keywords: {skip_keywords}')

    parameters = get_pretrain_param_groups(model, logger, skip, skip_keywords)

    opt_lower = config['training']['optimizer']
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config['training']['momentum'], nesterov=True,
                              lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config['training']['eps'], betas=config['training']['betas'],
                                lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])

    logger.info(f'>>>>>>>>>> Created {opt_lower} Optimizer for Pre-training Stage')

    return optimizer

def get_pretrain_param_groups(model, logger, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    logger.info(f'No decay params: {no_decay_name}')
    logger.info(f'Has decay params: {has_decay_name}')
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin