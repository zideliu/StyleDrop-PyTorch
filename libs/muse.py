import numpy as np
import torch
import math
from einops import rearrange
from torch.nn import functional as F


def add_gumbel_noise(t, temperature, device):
    return (t + torch.Tensor(temperature * np.random.gumbel(size=t.shape)).to(device))


class MUSE(object):
    def __init__(self, codebook_size, device, ignore_ind=-1, smoothing=0., gen_temp=4.5):
        self.mask_ind = codebook_size  # for input masking
        self.ignore_ind = ignore_ind  # for ce loss, excluding visible
        self.device = device
        self.smoothing = smoothing
        self.gen_temp = gen_temp

    @staticmethod
    def cosine_schedule(t):
        return torch.cos(t * math.pi * 0.5)

    def sample(self, x0):
        N, L, device = *x0.shape, self.device
        timesteps = torch.zeros((N,), device=device).float().uniform_(0, 1)
        rand_mask_probs = self.cosine_schedule(timesteps)  # cosine schedule
        num_token_masked = (L * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand(N, L, device=device).argsort(dim=-1)
        mask = batch_randperm < rearrange(num_token_masked, 'b -> b 1')
        masked_ids = torch.where(mask, self.mask_ind, x0)
        labels = torch.where(mask, x0, self.ignore_ind)
        return labels, masked_ids

    def loss(self, pred, label):
        return F.cross_entropy(pred.transpose(1, 2), label.long(),
                               ignore_index=self.ignore_ind, label_smoothing=self.smoothing)

    @torch.no_grad()
    def generate(self, config, _n_samples, nnet, decode_fn, is_eval=False, **kwargs):
        fmap_size, _sample_steps, device = config.z_shape[-1], config.sample.sample_steps, self.device

        seq_len = fmap_size ** 2
        ids = torch.full((_n_samples, seq_len), self.mask_ind, dtype=torch.long, device=device)
        cfg_scale = 0.
        for step in range(_sample_steps):
            ratio = 1. * (step + 1) / _sample_steps
            annealed_temp = self.gen_temp * (1 - ratio)
            is_mask = (ids == self.mask_ind)
            logits = nnet(ids, **kwargs, scale=cfg_scale)
            # sampling & scoring
            sampled_ids = add_gumbel_noise(logits, annealed_temp, device).argmax(dim=-1)
            sampled_logits = torch.squeeze(
                torch.gather(logits, dim=-1, index=torch.unsqueeze(sampled_ids, -1)), -1)
            sampled_ids = torch.where(is_mask, sampled_ids, ids)
            sampled_logits = torch.where(is_mask, sampled_logits, +np.inf).float()
            # masking
            mask_ratio = np.cos(ratio * math.pi * 0.5)
            mask_len = torch.Tensor([np.floor(seq_len * mask_ratio)]).to(device)
            mask_len = torch.maximum(torch.Tensor([1]).to(device),
                                     torch.minimum(torch.sum(is_mask, dim=-1, keepdims=True) - 1,
                                                   mask_len))[0].squeeze()
            confidence = add_gumbel_noise(sampled_logits, annealed_temp, device)
            sorted_confidence, _ = torch.sort(confidence, axis=-1)
            cut_off = sorted_confidence[:, mask_len.long() - 1:mask_len.long()]
            masking = (confidence <= cut_off)
            ids = torch.where(masking, self.mask_ind, sampled_ids)
            cfg_scale = ratio * config.sample.scale
            
        _z1 = rearrange(sampled_ids, 'b (i j) -> b i j', i=fmap_size, j=fmap_size)
        
        # with adapter
        ids = torch.full((_n_samples, seq_len), self.mask_ind, dtype=torch.long, device=device)
        cfg_scale = 0.
        lambdaA=0.
        lambdaB=0.
        for step in range(_sample_steps):
            ratio = 1. * (step + 1) / _sample_steps
            annealed_temp = self.gen_temp * (1 - ratio)
            is_mask = (ids == self.mask_ind)
            # 尝试使用 *ratio
            logits = nnet(ids, **kwargs, scale=cfg_scale,lambdaA=lambdaA,lambdaB=lambdaB)
            # sampling & scoring
            sampled_ids = add_gumbel_noise(logits, annealed_temp, device).argmax(dim=-1)
            sampled_logits = torch.squeeze(
                torch.gather(logits, dim=-1, index=torch.unsqueeze(sampled_ids, -1)), -1)
            sampled_ids = torch.where(is_mask, sampled_ids, ids)
            sampled_logits = torch.where(is_mask, sampled_logits, +np.inf).float()
            # masking
            mask_ratio = np.cos(ratio * math.pi * 0.5)
            mask_len = torch.Tensor([np.floor(seq_len * mask_ratio)]).to(device)
            mask_len = torch.maximum(torch.Tensor([1]).to(device),
                                     torch.minimum(torch.sum(is_mask, dim=-1, keepdims=True) - 1,
                                                   mask_len))[0].squeeze()
            confidence = add_gumbel_noise(sampled_logits, annealed_temp, device)
            sorted_confidence, _ = torch.sort(confidence, axis=-1)
            cut_off = sorted_confidence[:, mask_len.long() - 1:mask_len.long()]
            masking = (confidence <= cut_off)
            ids = torch.where(masking, self.mask_ind, sampled_ids)
            cfg_scale = ratio * config.sample.scale
            lambdaA = config.sample.lambdaA
            lambdaB = config.sample.lambdaB
        
        _z2 = rearrange(sampled_ids, 'b (i j) -> b i j', i=fmap_size, j=fmap_size)
        _z = _z2 if is_eval else torch.cat([_z1,_z2],dim=0) 
        out = decode_fn(_z)
        return out
