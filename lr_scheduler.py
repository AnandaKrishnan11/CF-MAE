# --------------------------------------------------------
# Based from SimMIM codebase
# https://github.com/microsoft/SimMIM
# --------------------------------------------------------

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler


def build_scheduler(config, optimizer, n_iter_per_epoch):

    num_steps = int(config['training']['epochs'] * n_iter_per_epoch)
    warmup_steps = int(config['training']['warmup_epochs'] * n_iter_per_epoch)
    decay_steps = int(config['training']['step_size'] * n_iter_per_epoch)
    
    config = config(is_pretrain=True)

    if config['training']['optimizer'] == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            t_mul=1.,
            lr_min=config['training']['min_lr'],
            warmup_lr_init=config['training']['warmup_lr'],
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif config['training']['optimizer'] == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config['training']['gamma'],
            warmup_lr_init=config['training']['warmup_lr'],
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    return lr_scheduler

