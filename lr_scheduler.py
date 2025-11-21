# --------------------------------------------------------
# Based from SimMIM codebase
# https://github.com/microsoft/SimMIM
# --------------------------------------------------------

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler


def build_scheduler(config, logger, optimizer, n_iter_per_epoch):

    num_steps = int(config['TRAINING']['TOTAL_EPOCHS'] * n_iter_per_epoch)
    warmup_steps = int(config['TRAINING']['WARMUP_EPOCHS'] * n_iter_per_epoch)
    decay_steps = int(config['TRAINING']['STEP_SIZE'] * n_iter_per_epoch)
    
    logger.info('>>>>>>>>>>>>>>>> Building Scheduler for Pre-training Stage')

    if config['TRAINING']['OPTIMIZER'] == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            t_mul=1.,
            lr_min=config['TRAINING']['MIN_LR'],
            warmup_lr_init=config['TRAINING']['WARMUP_LR'],
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif config['TRAINING']['OPTIMIZER'] == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config['TRAINING']['GAMMA'],
            warmup_lr_init=config['TRAINING']['WARMUP_LR'],
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    logger.info(f'>>>>>>>>>>>>>>>> Created {config['TRAINING']['OPTIMIZER']} Scheduler for Pre-training Stage')
    return lr_scheduler

