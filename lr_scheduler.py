# --------------------------------------------------------
# Based from SimMIM codebase
# https://github.com/microsoft/SimMIM
# --------------------------------------------------------

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_scheduler(config, logger, optimizer, n_iter_per_epoch):

    num_steps = int(config['TRAINING']['TOTAL_EPOCHS'] * n_iter_per_epoch)
    warmup_steps = int(config['TRAINING']['WARMUP_EPOCHS'] * n_iter_per_epoch)
    decay_steps = int(config['TRAINING']['STEP_SIZE'] * n_iter_per_epoch)
    
    logger.info('>>>>>>>>>>>>>>>> Building Scheduler for Pre-training Stage')

    if config['TRAINING']['SCHEDULER'] == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=float(config['TRAINING']['MIN_LR']),
            warmup_lr_init=float(config['TRAINING']['WARMUP_LR']),
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif config['TRAINING']['SCHEDULER'] == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=float(config['TRAINING']['GAMMA']),
            warmup_lr_init=float(config['TRAINING']['WARMUP_LR']),
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config['TRAINING']['SCHEDULER'] == 'reducelr':
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            threshold=1e-3,
            threshold_mode='rel',
            cooldown=1,
            patience=1,
            min_lr=float(config['TRAINING']['MIN_LR'])
        )
    else:
        raise NotImplementedError(f"Scheduler {config['TRAINING']['SCHEDULER']} not implemented")
    
    logger.info('  Initial learning rates per group:')
    for i, group in enumerate(optimizer.param_groups):
        group_name = group.get('name', f'group_{i}')
        logger.info(f'    • {group_name}: {group["lr"]:.2e}')

    return lr_scheduler

