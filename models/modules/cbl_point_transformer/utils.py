import re


def fetch_pxo(stage_n, stage_i, stage_list, ftype):
    stage = stage_list[stage_n][stage_i]
    return stage['p_out'], stage[ftype], stage['offset']

def fetch_supports_stage(inputs, stage_n, stage_i, ftype):
    # indexing the existing stages - all built
    stage_n = to_valid_stage(stage_n)
    stage = inputs['stage_list'][stage_n][stage_i]
    ftype, ptype = get_ftype(ftype)
    pts = stage[ptype]
    f = stage[ftype]
    idx = inputs['neighbors'][stage_i]
    return pts, f, idx

def to_valid_stage(stage_n, short=False):
    if stage_n in ['D', 'down']:
        stage_n = 'D' if short else 'down'
    elif stage_n in ['U', 'up']:
        stage_n = 'U' if short else 'up'
    else:
        raise ValueError(f'invalid stage_n={stage_n}')
    return stage_n

def parse_stage(stage, num_layers):
    stage = stage.replace('a', ''.join(f'{i}' for i in range(num_layers)))
    stage_list = [i.strip('_') for i in re.split('(\d+)', stage) if i and i.strip('_')]  # e.g. D012_U34
    assert len(stage_list) % 2 == 0, f'invalid stage compound: stage_list={stage_list} from stage={stage}'
    stage_n = [s for i, s in enumerate(stage_list) if i % 2 == 0]
    stage_i = [s for i, s in enumerate(stage_list) if i % 2 == 1]
    stage_list = [[(to_valid_stage(n), int(i)) for i in i_str] for n, i_str in zip(stage_n, stage_i)]
    stage_list = sum(stage_list, [])
    return stage_list


def get_kr(config, stage_n, stage_i):
    assert stage_n in _valid_stage, f'invalid stage_n={stage_n}'
    if stage_n:
        kr = config.kr_sample[stage_i - 1] if stage_n == 'down' else config.kr_sample_up[stage_i]
    else:
        kr = config.kr_search[stage_i]
    return kr

def get_kwargs(block_cfg, config, is_training, act=False):
    kwargs = {
        'is_training': is_training,
        'initializer': block_cfg.init if block_cfg.init else config.init,
        'weight_decay': block_cfg.wd if block_cfg.wd else config.weight_decay,
        'bn_momentum': config.bn_momentum, 'bn_eps': config.bn_eps
    }
    if act is True:
        kwargs['activation'] = block_cfg.act if block_cfg.act else config.activation
    elif act:
        kwargs['activation'] = act
    return kwargs

def get_ftype(ftype):
    if ftype in ['out', 'fout', 'f_out', 'latent', 'logits', 'probs']:
        ptype = 'p_out'
        ftype = 'f_out' if ftype in ['out', 'fout'] else ftype
    elif ftype in ['sample', 'fsample', 'f_sample']:
        ptype = 'p_sample'
        ftype = 'f_sample' if ftype in ['sample', 'fsample'] else ftype
    else:
        raise KeyError(f'not supported ftype = {ftype}')
    return ftype, ptype


def traverse_list(l, fn, inplace=True, try_fn=True):
    container = l if inplace else []
    for i, v in enumerate(l):
        if isinstance(v, dict):
            v = traverse_dict(v, fn, inplace, try_fn)
        elif isinstance(v, list):
            v = traverse_list(v, fn, inplace, try_fn)
        elif try_fn:
            try: v = fn(v)
            except: pass
        else:
            v = fn(v)
        container[i] = v
    return container

def traverse_dict(d, fn, inplace=True, try_fn=True):
    container = d if inplace else {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = traverse_dict(v, fn, inplace, try_fn)
        elif isinstance(v, list):
            v = traverse_list(v, fn, inplace, try_fn)
        elif try_fn:
            try: v = fn(v)
            except: pass
        else:
            v = fn(v)
        container[k] = v
    return container

class TorchDict(dict):
    def __init__(self, *args, **kwargs):
        super(TorchDict, self).__init__(*args, **kwargs)

    def cpu(self, inplace=True, try_fn=True):
        return traverse_dict(self, torch.Tensor.cpu, inplace=inplace, try_fn=try_fn)

    def numpy(self, inplace=True, try_fn=True):
        return traverse_dict(self, lambda x: x.cpu().numpy(), inplace=inplace, try_fn=try_fn)

    def to(self, device, inplace=True, try_fn=True):
        return traverse_dict(self, lambda x, d=device: x.to(d), inplace=inplace, try_fn=try_fn)

    def cuda(self, inplace=True, try_fn=True):
        return traverse_dict(self, lambda x: x.cuda(), inplace=inplace, try_fn=try_fn)

class TorchList(list):
    def __init__(self, *args, **kwargs):
        super(TorchList, self).__init__(*args, **kwargs)

    def cpu(self, inplace=True, try_fn=True):
        return traverse_list(self, torch.Tensor.cpu, inplace=inplace, try_fn=try_fn)

    def numpy(self, inplace=True, try_fn=True):
        return traverse_list(self, lambda x: x.cpu().numpy(), inplace=inplace, try_fn=try_fn)

    def to(self, device, inplace=True, try_fn=True):
        return traverse_list(self, lambda x, d=device: x.to(d), inplace=inplace, try_fn=try_fn)

    def cuda(self, inplace=True, try_fn=True):
        return traverse_list(self, lambda x: x.cuda(), inplace=inplace, try_fn=try_fn)
