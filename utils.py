import os
import torch

# Pretrained change encoder: [Linear(0), GELU(1), Linear(2)]
# Fine-tuning change encoder: [Linear(0), GELU(1), Dropout(2), Linear(3), Dropout(4)]
CHANGE_ENCODER_INDEX_MAP = {"0": "0", "2": "3"}

SKIP_PREFIXES = ("mask_token", "decoder.")


def read_state_dict(path, weights_only=True):
    if not os.path.isfile(path):
        raise FileNotFoundError("checkpoint not found: {}".format(path))

    obj = torch.load(path, map_location="cpu", weights_only=weights_only)
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

    state = dict(obj)
    print("Loaded {} tensors from {}".format(len(state), path))
    return state


def split_by_prefix(state, prefixes):
    groups = {p: {} for p in prefixes}
    leftover = {}

    for key, value in state.items():
        if key.startswith(SKIP_PREFIXES):
            continue
        match = None
        for prefix in prefixes:
            if key.startswith(prefix) and (match is None or len(prefix) > len(match)):
                match = prefix
        if match is None:
            leftover[key] = value
        else:
            groups[match][key[len(match):]] = value

    if leftover:
        print("[warn] ignoring {} unrecognised keys, e.g. {}".format(
            len(leftover), sorted(leftover)[:5]))
    return groups


def remap_layer_indices(state, index_map):
    remapped = {}
    for key, value in state.items():
        index, _, rest = key.partition(".")
        if index in index_map:
            remapped[index_map[index] + "." + rest] = value
    return remapped


def load_into(module, state, name, strict=False):
    if not state:
        raise KeyError("no pretrained weights found for '{}'".format(name))

    missing, unexpected = module.load_state_dict(state, strict=strict)
    loaded = len(state) - len(unexpected)
    if loaded == 0:
        raise KeyError("no checkpoint key matched '{}' (check the prefixes)".format(name))

    print("{}: loaded {}/{} tensors (missing={}, unexpected={})".format(
        name, loaded, len(module.state_dict()), len(missing), len(unexpected)))
    if unexpected:
        print("[warn] {} unexpected keys: {}".format(name, unexpected))


def load_v1(model, state):
    prefixes = ("encoder.", "change_encoder.", "decoder_embed.",
                "pre_projection.", "post_projection.")
    groups = split_by_prefix(state, prefixes)
    target = model.encoderv1

    load_into(target.encoder, groups["encoder."], "encoder", strict=True)
    load_into(target.pre_projection, groups["pre_projection."], "pre_projection", strict=True)
    load_into(target.post_projection, groups["post_projection."], "post_projection", strict=True)
    load_into(target.change_encoder, groups["change_encoder."], "change_encoder", strict=True)
    load_into(target.decoder_embed, groups["decoder_embed."], "decoder_embed", strict=True)


def load_decoder_embed(module, state):
    """Copy the pretrained Linear into the first/last Linear of the Sequential.

    v2: decoder_embed is D -> D ... D -> D, so both layer 0 and layer 3 match.
    v3: decoder_embed fuses 4 scales (4*D -> 2*D -> D), nothing matches and the
    module keeps its default init.
    """
    target = module.state_dict()
    expanded = {}
    for key, value in state.items():
        for index in ("0", "3"):
            name = index + "." + key
            if name in target and target[name].shape == value.shape:
                expanded[name] = value

    if not expanded:
        print("decoder_embed: shapes differ from the checkpoint, keeping default init")
        return
    load_into(module, expanded, "decoder_embed")


def load_multiscale(model, state):
    """Shared loading path for v2 and v3."""
    groups = split_by_prefix(state, ("encoder.", "change_encoder.", "decoder_embed."))

    load_into(model.encoder.encoder, groups["encoder."], "encoder")

    change_state = remap_layer_indices(groups["change_encoder."], CHANGE_ENCODER_INDEX_MAP)
    for i in range(1, 5):
        branch = getattr(model.encoder, "change_encoder{}".format(i))
        load_into(branch, change_state, "change_encoder{}".format(i))

    load_decoder_embed(model.encoder.decoder_embed, groups["decoder_embed."])


def load_pretrained_cfmae(model, config):
    version = config["VERSION"]
    state = read_state_dict(config["TRAINING"]["PRETRAINED_WEIGHT"])
    print("Initialising {} from CF-MAE checkpoint".format(version))

    if version == "v1":
        load_v1(model, state)
    elif version in ("v2", "v3"):
        load_multiscale(model, state)
    else:
        raise ValueError("unknown model version: {}".format(version))
    return model