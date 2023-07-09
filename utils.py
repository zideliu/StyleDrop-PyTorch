import pickle

import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from torchvision.utils import save_image
from torch import distributed as dist
from loguru import logger
logging = logger


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


def get_nnet(name, **kwargs):
    if name == 'uvit_t2i_vq':
        from libs.uvit_t2i_vq import UViT
        return UViT(**kwargs)
    elif name == 'uvit_vq':
        from libs.uvit_vq import UViT
        return UViT(**kwargs)
    else:
        raise NotImplementedError(name)


def set_seed(seed: int):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


def get_optimizer(params, name, **kwargs):
    if name == 'adam':
        from torch.optim import Adam
        return Adam(params, **kwargs)
    elif name == 'adamw':
        from torch.optim import AdamW
        return AdamW(params, **kwargs)
    else:
        raise NotImplementedError(name)


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        if 'adapter' not in p_name:
            p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)
        else:
            p_dest.data = p_src.detach().clone()


class TrainState(object):
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path, adapter_only=False,name=""):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        if adapter_only:
            torch.save(self.nnet.adapter.state_dict(), os.path.join(path, name+'adapter.pth'))
        else:
            
            for key, val in self.__dict__.items():
                if key != 'step' and val is not None:
                    torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))
                
    def make_dict(self,model,state_dict):
        state = {}
        for k in model.state_dict().keys():
            if k in state_dict:
                state[k] = state_dict[k].clone()
            else:
                state[k] = model.state_dict()[k].clone()
        return state
    
    def load(self, path):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'), map_location='cpu')
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None and key != 'optimizer' and key != 'lr_scheduler':
                if key == 'nnet' or key == 'nnet_ema':
                    val.load_state_dict(self.make_dict(val,torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu')))
                else:
                    val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))
                    
    def load_adapter(self,path):
        logging.info('load adapter from {}'.format(path))
        adapter = torch.load(path,map_location='cpu')
        keys=['nnet','nnet_ema']
        for key in keys:
            if key in self.__dict__:
                self.__dict__[key].adapter.load_state_dict(adapter)
            else:
                logging.info('adapter not in state_dict')
                
    def resume(self, ckpt_root,adapter_path=None, step=None):
        if not os.path.exists(ckpt_root):
            return
        if ckpt_root.endswith('.ckpt'):
            ckpt_path = ckpt_root
        else:
            if step is None:
                ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
                if not ckpts:
                    return
                steps = map(lambda x: int(x.split(".")[0]), ckpts)
                step = max(steps)
            ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path)
        if adapter_path is not None:
            self.load_adapter(adapter_path)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)
    def freeze(self):
        self.nnet.requires_grad_(False)
        for name, p in self.nnet.named_parameters():
            if 'adapter' in name:
                p.requires_grad_(True)
                

def cnt_params(model):
    return sum(param.numel() for param in model.parameters())


def initialize_train_state(config, device):
    params = []

    nnet = get_nnet(**config.nnet)
    params += nnet.adapter.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.to(device)
    return train_state


def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def sample2dir(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None, dist=True):
    if path:
        os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes if dist else mini_batch_size

    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
        samples = unpreprocess_fn(sample_fn(mini_batch_size))
        if dist:
            samples = accelerator.gather(samples.contiguous())[:_batch_size]
        if accelerator.is_main_process:
            for sample in samples:
                save_image(sample, os.path.join(path, f"{idx}.png"))
                idx += 1


def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

from collections import defaultdict, deque
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter=" "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    from torch import inf
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

