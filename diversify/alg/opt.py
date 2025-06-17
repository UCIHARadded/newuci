import torch

def get_params(alg, args, nettype):
    init_lr = args.lr

    def safe_param_block(name, module, lr_mult):
        if module is not None:
            return {'params': module.parameters(), 'lr': lr_mult * init_lr}
        else:
            print(f"[WARNING] {name} is None in get_params ({nettype}) — skipping.")
            return None

    if nettype == 'Diversify-adv':
        blocks = [
            safe_param_block("dbottleneck", alg.dbottleneck, args.lr_decay2),
            safe_param_block("dclassifier", alg.dclassifier, args.lr_decay2),
            safe_param_block("ddiscriminator", alg.ddiscriminator, args.lr_decay2)
        ]
    elif nettype == 'Diversify-cls':
        blocks = [
            safe_param_block("bottleneck", alg.bottleneck, args.lr_decay2),
            safe_param_block("classifier", alg.classifier, args.lr_decay2),
            safe_param_block("discriminator", alg.discriminator, args.lr_decay2)
        ]
    elif nettype == 'Diversify-all':
        blocks = [
            safe_param_block("featurizer", alg.featurizer, args.lr_decay1),
            safe_param_block("abottleneck", alg.abottleneck, args.lr_decay2),
            safe_param_block("aclassifier", alg.aclassifier, args.lr_decay2)
        ]
    else:
        raise ValueError(f"Unknown nettype: {nettype}")

    # Filter out None blocks (in case something wasn't initialized yet)
    return [block for block in blocks if block is not None]

def get_optimizer(alg, args, nettype):
    params = get_params(alg, args, nettype=nettype)
    return torch.optim.Adam(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, 0.9)
    )
