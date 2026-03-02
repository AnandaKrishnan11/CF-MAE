# --------------------------------------------------------
# Based from SimMIM codebase
# https://github.com/microsoft/SimMIM
# --------------------------------------------------------

from torch import optim as optim
import torch


def build_optimizer(config, model, logger, is_pretrain=True):

    if is_pretrain: 
        return build_pretrain_optimizer(config, model, logger)
    else:
        return build_finetune_optimizer(config, model, logger) 


def build_finetune_optimizer(config, model, logger): 
    """
    Create optimizer with layer-wise learning rates for segmentation finetuning
    """
    logger.info('>>>>>>>>>> Build Optimizer for Finetuning Stage')
    
    base_lr = float(config['TRAINING']['LEARNING_RATE'])
    weight_decay = float(config['TRAINING']['WEIGHT_DECAY'])
    
    model_version = config['VERSION']

    encoder_lr_mult = config['TRAINING'].get('ENCODER_LR_MULT', 0.1)
    change_encoder_lr_mult = config['TRAINING'].get('CHANGE_ENCODER_LR_MULT', 0.5)
    decoder_embed_lr_mult = config['TRAINING'].get('DECODER_EMBED_LR_MULT', 0.5)
    decoder_lr_mult = config['TRAINING'].get('DECODER_LR_MULT', 1.0)

    if hasattr(model,'no_weight_decay'):
            skip_keyword = model.no_weight_decay()

    if model_version =="v1":

        #Encoder
        encoder_params = []
        encoder_no_decay = []
        for name,param in model.encoderv1.encoder.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name or skip_keyword in name:
                encoder_no_decay.append(param)
            else:
                encoder_params.append(param)

        logger.info(f'ENCODER WITH DECAY--{encoder_params}')

        #Single Change encoder 
        change_encoder_params = []
        change_encoder_no_decay = []       
        for name, param in model.encoderv1.change_encoder.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias"):
                change_encoder_no_decay.append(param)
            else:
                change_encoder_params.append(param)

        logger.info(f'CHANGE ENCODER WITH DECAY--{change_encoder_params}')

        #Decoder embed
        decoder_embed_params = []
        decoder_embed_no_decay = []
        for name, param in model.encoderv1.decoder_embed.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias"):
                decoder_embed_no_decay.append(param)
            else:
                decoder_embed_params.append(param)

        logger.info(f'MLP PROJ WITH DECAY--{decoder_embed_params}')
        
        #Segmentation decoder
        decoder_params = []
        decoder_no_decay = []
        for name, param in model.decoderv1.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name:
                decoder_no_decay.append(param)
            else:
                decoder_params.append(param)

        logger.info(f'DECODER WITH DECAY--{decoder_params}')
        
        param_groups = [
            {
                'params': encoder_params,
                'lr': base_lr * encoder_lr_mult,
                'weight_decay': weight_decay,
                'name': 'encoder_backbone'
            },
            {
                'params': encoder_no_decay,
                'lr': base_lr * encoder_lr_mult,
                'weight_decay': 0.0,
                'name': 'encoder_backbone_no_decay'
            },
            {
                'params': change_encoder_params,
                'lr': base_lr * change_encoder_lr_mult,
                'weight_decay': weight_decay,
                'name': 'change_encoders'
            },
            {
                'params': change_encoder_no_decay,
                'lr': base_lr * change_encoder_lr_mult,
                'weight_decay': 0.0,
                'name': 'change_encoders_no_decay'
            },
            {
                'params': decoder_embed_params,
                'lr': base_lr * decoder_embed_lr_mult,
                'weight_decay': weight_decay,
                'name': 'decoder_embed'
            },
            {
                'params': decoder_embed_no_decay,
                'lr': base_lr * decoder_embed_lr_mult,
                'weight_decay': 0.0,
                'name': 'decoder_embed_no_decay'
            },
            {
                'params': decoder_params,
                'lr': base_lr * decoder_lr_mult,
                'weight_decay': weight_decay,
                'name': 'segmentation_decoder'
            },
            {
                'params': decoder_no_decay,
                'lr': base_lr * decoder_lr_mult,
                'weight_decay': 0.0,
                'name': 'segmentation_decoder_no_decay'
            }
        ]
            
    else:

        #Encoder
        encoder_params = []
        encoder_no_decay = []
        for name, param in model.encoder.encoder.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name or skip_keyword in name:
                encoder_no_decay.append(param)
            else:
                encoder_params.append(param)

        logger.info(f'ENCODER WITH DECAY--{encoder_params}')
        
        
        #Change encoders 1-4 
        change_encoder_params = []
        change_encoder_no_decay = []
        for i in range(1, 5):
            ce = getattr(model.encoder, f'change_encoder{i}')
            for name, param in ce.named_parameters():
                if not param.requires_grad:
                    continue
                if len(param.shape) == 1 or name.endswith(".bias"):
                    change_encoder_no_decay.append(param)
                else:
                    change_encoder_params.append(param)

        logger.info(f'CHANGE ENCODER WITH DECAY--{change_encoder_params}')
        
        #Decoder embed
        decoder_embed_params = []
        decoder_embed_no_decay = []
        for name, param in model.encoder.decoder_embed.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias"):
                decoder_embed_no_decay.append(param)
            else:
                decoder_embed_params.append(param)

        logger.info(f'MLP PROJ WITH DECAY--{decoder_embed_params}')
        
        #Segmentation decoder
        decoder_params = []
        decoder_no_decay = []
        for name, param in model.decoder.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name:
                decoder_no_decay.append(param)
            else:
                decoder_params.append(param)

        logger.info(f'DECODER WITH DECAY--{decoder_params}')
        
        param_groups = [
            {
                'params': encoder_params,
                'lr': base_lr * encoder_lr_mult,
                'weight_decay': weight_decay,
                'name': 'encoder_backbone'
            },
            {
                'params': encoder_no_decay,
                'lr': base_lr * encoder_lr_mult,
                'weight_decay': 0.0,
                'name': 'encoder_backbone_no_decay'
            },
            {
                'params': change_encoder_params,
                'lr': base_lr * change_encoder_lr_mult,
                'weight_decay': weight_decay,
                'name': 'change_encoders'
            },
            {
                'params': change_encoder_no_decay,
                'lr': base_lr * change_encoder_lr_mult,
                'weight_decay': 0.0,
                'name': 'change_encoders_no_decay'
            },
            {
                'params': decoder_embed_params,
                'lr': base_lr * decoder_embed_lr_mult,
                'weight_decay': weight_decay,
                'name': 'decoder_embed'
            },
            {
                'params': decoder_embed_no_decay,
                'lr': base_lr * decoder_embed_lr_mult,
                'weight_decay': 0.0,
                'name': 'decoder_embed_no_decay'
            },
            {
                'params': decoder_params,
                'lr': base_lr * decoder_lr_mult,
                'weight_decay': weight_decay,
                'name': 'segmentation_decoder'
            },
            {
                'params': decoder_no_decay,
                'lr': base_lr * decoder_lr_mult,
                'weight_decay': 0.0,
                'name': 'segmentation_decoder_no_decay'
            }
        ]
        
    # Create optimizer
    opt_lower = config['TRAINING']['OPTIMIZER'].lower()
    
    if opt_lower == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            eps=float(config['TRAINING'].get('EPS', 1e-8)),
            betas=tuple(config['TRAINING'].get('BETAS', (0.9, 0.999)))
        )
    elif opt_lower == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=config['TRAINING'].get('MOMENTUM', 0.9),
            nesterov=True
        )
    else:
        raise NotImplementedError(f"Optimizer {opt_lower} not implemented")
    
    # Log parameter groups
    logger.info("\n[Optimizer] Parameter groups:")
    total_params = 0
    for group in param_groups:
        n_params = sum(p.numel() for p in group['params'])
        total_params += n_params
        if n_params > 0:  # Only log non-empty groups
            wd_str = f", WD={group['weight_decay']:.2e}" if group['weight_decay'] > 0 else ", no WD"
            logger.info(f"  • {group['name']}: {n_params:,} params, LR={group['lr']:.2e}{wd_str}")
    
    logger.info(f"\nTotal trainable parameters: {total_params:,}")
    logger.info(f'>>>>>>>>>> Created {opt_lower.upper()} Optimizer for Finetuning Stage')
    
    return optimizer


def build_pretrain_optimizer(config, model, logger):
    """
    Build optimizer for pretraining stage (MAE-style)
    """
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

    opt_lower = config['TRAINING']['OPTIMIZER'].lower()
    
    if opt_lower == 'sgd':
        optimizer = optim.SGD(
            parameters, 
            momentum=config['TRAINING']['MOMENTUM'], 
            nesterov=True,
            lr=float(config['TRAINING']['LEARNING_RATE']), 
            weight_decay=config['TRAINING']['WEIGHT_DECAY']
        )
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(
            parameters, 
            eps=float(config['TRAINING'].get('EPS', 1e-8)), 
            betas=tuple(config['TRAINING'].get('BETAS', (0.9, 0.999))),
            lr=float(config['TRAINING']['LEARNING_RATE']), 
            weight_decay=config['TRAINING']['WEIGHT_DECAY']
        )
    else:
        raise NotImplementedError(f"Optimizer {opt_lower} not implemented")

    logger.info(f'>>>>>>>>>> Created {opt_lower.upper()} Optimizer for Pre-training Stage')
    return optimizer


def get_pretrain_param_groups(model, logger, skip_list=(), skip_keywords=()):
    """
    Separate parameters into groups with and without weight decay
    """
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay for: 1D params (bias, norms), params in skip_list, or matching skip_keywords
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    
    logger.info(f'Parameters with weight decay: {len(has_decay_name)} groups')
    logger.info(f'Parameters without weight decay: {len(no_decay_name)} groups')
    
    # Log first few of each for verification
    if has_decay_name:
        logger.info(f'  Examples with decay: {has_decay_name[:3]}...')
    if no_decay_name:
        logger.info(f'  Examples without decay: {no_decay_name[:3]}...')
    
    return [
        {'params': has_decay},
        {'params': no_decay, 'weight_decay': 0.}
    ]


def check_keywords_in_name(name, keywords=()):
    """
    Check if any keyword appears in the parameter name
    """
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin