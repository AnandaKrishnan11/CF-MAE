import torch

NO_DECAY_KEYWORDS = ("pos_embed", "cls_token", "norm")

LR_RULES = (
    ("encoder.encoder.", "ENCODER_LR_MULT", 0.1, "encoder_backbone"),
    ("encoder.change_encoder", "CHANGE_ENCODER_LR_MULT", 0.5, "change_encoder"),
    ("encoder.decoder_embed", "DECODER_EMBED_LR_MULT", 0.5, "decoder_embed"),
    ("encoder.", "CHANGE_ENCODER_LR_MULT", 0.5, "encoder_head"),  # v1 pre/post projection
    ("decoder.", "DECODER_LR_MULT", 1.0, "segmentation_decoder"),
)


def build_optimizer(config, model, logger, is_pretrain=True):
    if is_pretrain:
        return build_pretrain_optimizer(config, model, logger)
    return build_finetune_optimizer(config, model, logger)


def is_no_decay(name, param, extra_keywords=()):
    if param.ndim <= 1 or name.endswith(".bias"):
        return True
    keywords = tuple(NO_DECAY_KEYWORDS) + tuple(extra_keywords)
    return any(keyword in name for keyword in keywords)


def model_keywords(model):
    keywords = []
    for attr in ("no_weight_decay", "no_weight_decay_keywords"):
        if hasattr(model, attr):
            value = getattr(model, attr)()
            keywords.extend([value] if isinstance(value, str) else list(value))
    return keywords

def match_rule(config, name):
    for prefix, key, default, group in LR_RULES:
        if name.startswith(prefix):
            return group, float(config["TRAINING"].get(key, default))
    return "other", 1.0

def build_finetune_optimizer(config, model, logger):
    """Layer-wise learning rates for segmentation finetuning (all versions)."""
    logger.info(">>>>>>>>>> Build Optimizer for Finetuning Stage")

    base_lr = float(config["TRAINING"]["LEARNING_RATE"])
    weight_decay = float(config["TRAINING"]["WEIGHT_DECAY"])
    keywords = model_keywords(model)

    buckets = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group, mult = match_rule(config, name)
        if group == "other":
            logger.warning(f"{name} matched no LR rule, using the base learning rate")
        decay = not is_no_decay(name, param, keywords)
        buckets.setdefault((group, decay), {"lr": base_lr * mult, "params": []})
        buckets[(group, decay)]["params"].append(param)

    param_groups = []
    for (group, decay), bucket in buckets.items():
        param_groups.append({
            "params": bucket["params"],
            "lr": bucket["lr"],
            "weight_decay": weight_decay if decay else 0.0,
            "name": group if decay else group + "_no_decay",
        })

    optimizer = create_optimizer(config, param_groups, logger)
    log_param_groups(model, param_groups, logger)
    logger.info(">>>>>>>>>> Created optimizer for Finetuning Stage")
    return optimizer

def build_pretrain_optimizer(config, model, logger):
    """Two groups only (decay / no decay), MAE style."""
    logger.info(">>>>>>>>>> Build Optimizer for Pre-training Stage")

    base_lr = float(config["TRAINING"]["LEARNING_RATE"])
    weight_decay = float(config["TRAINING"]["WEIGHT_DECAY"])
    keywords = model_keywords(model)
    logger.info(f"No weight decay keywords: {keywords}")

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        target = no_decay if is_no_decay(name, param, keywords) else decay
        target.append(param)

    param_groups = [
        {"params": decay, "lr": base_lr, "weight_decay": weight_decay, "name": "decay"},
        {"params": no_decay, "lr": base_lr, "weight_decay": 0.0, "name": "no_decay"},
    ]

    optimizer = create_optimizer(config, param_groups, logger)
    log_param_groups(model, param_groups, logger)
    logger.info(">>>>>>>>>> Created optimizer for Pre-training Stage")
    return optimizer

def create_optimizer(config, param_groups, logger):
    """Instantiate the optimizer; every group already carries its own lr/wd."""
    param_groups = [g for g in param_groups if g["params"]]
    name = config["TRAINING"]["OPTIMIZER"].lower()

    if name == "adamw":
        return torch.optim.AdamW(
            param_groups,
            eps=float(config["TRAINING"].get("EPS", 1e-8)),
            betas=tuple(config["TRAINING"].get("BETAS", (0.9, 0.999))),
        )
    if name == "sgd":
        return torch.optim.SGD(
            param_groups,
            momentum=float(config["TRAINING"].get("MOMENTUM", 0.9)),
            nesterov=True,
        )
    raise NotImplementedError(f"Optimizer {name} not implemented")

def log_param_groups(model, param_groups, logger):
    """Log each group and check that no trainable parameter was left out."""
    logger.info("[Optimizer] Parameter groups:")
    total = 0
    for group in param_groups:
        n = sum(p.numel() for p in group["params"])
        if n == 0:
            continue
        total += n
        wd = f"WD={group['weight_decay']:.2e}" if group["weight_decay"] > 0 else "no WD"
        logger.info(f"  {group['name']}: {n:,} params, LR={group['lr']:.2e}, {wd}")

    expected = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters in the optimizer: {total:,} / {expected:,} trainable")
    if total != expected:
        raise RuntimeError(
            f"{expected - total:,} trainable parameters were not assigned to any group"
        )