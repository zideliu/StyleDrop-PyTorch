import torch
import torch.nn as nn
import math

from loguru import logger

import timm
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import PatchEmbed, Mlp

assert timm.__version__ == "0.3.2"  # version check
import einops
import torch.utils.checkpoint
import torch.nn.functional as F

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
    print("xformers available, will use xformers attention")
except:
    XFORMERS_IS_AVAILABLE = False
    print("xformers not available, will use pytorch attention instead")

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, vocab_size, hidden_size, max_position_embeddings, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

        torch.nn.init.normal_(self.word_embeddings.weight, std=.02)
        torch.nn.init.normal_(self.position_embeddings.weight, std=.02)

    def forward(
            self, input_ids
    ):
        input_shape = input_ids.size()

        seq_length = input_shape[1]

        position_ids = self.position_ids[:, :seq_length]

        inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MlmLayer(nn.Module):

    def __init__(self, feat_emb_dim, word_emb_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(feat_emb_dim, word_emb_dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(word_emb_dim)
        self.bias = nn.Parameter(torch.zeros(1, 1, vocab_size))

    def forward(self, x, word_embeddings):
        mlm_hidden = self.fc(x)
        mlm_hidden = self.gelu(mlm_hidden)
        mlm_hidden = self.ln(mlm_hidden)
        word_embeddings = word_embeddings.transpose(0, 1)
        logits = torch.matmul(mlm_hidden, word_embeddings)
        logits = logits + self.bias
        return logits


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        if XFORMERS_IS_AVAILABLE:
            qkv = self.qkv(x)
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Adapter(nn.Module):
    def __init__(self, d_emb:int, d_prj:int,n_layer: int, is_shared: bool):
        super().__init__()
        self.D = d_emb
        self.H = d_prj
        self.L = n_layer
        self.is_shared = is_shared
        if self.is_shared:
            self.DD = nn.Embedding(self.L,self.H)
            self.DU = nn.Embedding(self.L,self.D)
            self.WD = nn.Embedding(1,self.D*self.H)
            self.WU = nn.Embedding(1,self.H*self.D)
        else:
            self.WD = nn.Embedding(self.L,self.D*self.H)
            self.WU = nn.Embedding(self.L,self.H*self.D)
        self.activate = nn.GELU()

        self._init_weights()
    def _init_weights(self):
        for p in self.WU.parameters():
            p.detach().zero_()
        nn.init.trunc_normal_(self.WD.weight,mean=0,std=0.02)
        
        if self.is_shared:
            nn.init.trunc_normal_(self.DD.weight,mean=0,std=0.02)
            for p in self.DU.parameters():
                p.detach().zero_()
            
    def forward(self, emb, layer):
        idx = torch.arange(self.L).to(emb.device)
        layer = torch.tensor(layer).to(emb.device)
        if self.is_shared:
            idx0 = torch.zeros_like(idx).to(emb.device)
            dd = self.DD(idx).reshape(self.L, 1,self.H)
            du = self.DU(idx).reshape(self.L, 1,self.D)
            wd = self.WD(idx0).reshape(self.L, self.D,self.H) + dd
            wu = self.WU(idx0).reshape(self.L, self.H,self.D) + du
        else:
            wd = self.WD(idx).reshape(self.L, self.D,self.H)
            wu = self.WU(idx).reshape(self.L, self.H,self.D)
        
        prj = torch.einsum('...d,dh->...h',emb,wd[layer])
        prj = self.activate(prj)
        prj = torch.einsum('...h,hd->...d',prj,wu[layer])
        return emb + prj
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None, adapter=None, layer=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip, adapter, layer)
        else:
            return self._forward(x, skip, adapter, layer)

    def _forward(self, x, skip=None,adapter=None, layer=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
            
        attn = self.attn(self.norm1(x))
        if adapter is not None:
            attn = adapter(attn, layer)
            
        x = x + attn
        x = x + self.mlp(self.norm2(x))
        return x


class UViT(nn.Module):
    def __init__(self, img_size=16, in_chans=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, use_checkpoint=False,
                 clip_dim=768, num_clip_token=77, skip=True, codebook_size=1024,d_prj=4,is_shared=True):
        super().__init__()
        logger.debug(f'codebook size in nnet: {codebook_size}')
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.in_chans = in_chans
        self.skip = skip

        self.codebook_size = codebook_size
        vocab_size = codebook_size + 1
        self.time_embed = None
        self.extras = num_clip_token
        self.num_vis_tokens = int((img_size) ** 2)
        self.token_emb = BertEmbeddings(vocab_size=vocab_size,
                                        hidden_size=embed_dim,
                                        max_position_embeddings=self.num_vis_tokens,
                                        dropout=0.1)
        print(f'num vis tokens: {self.num_vis_tokens}')

        self.context_embed = nn.Linear(clip_dim, embed_dim)

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.mlm_layer = MlmLayer(feat_emb_dim=embed_dim, word_emb_dim=embed_dim, vocab_size=vocab_size)
        self.adapter = Adapter(d_emb=embed_dim, d_prj=d_prj, n_layer=depth, is_shared=is_shared)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore # type: ignore
    def no_weight_decay(self):
        return {'pos_embed'}
    
    def forward(self, masked_ids, context,use_adapter=False):
        assert len(masked_ids.shape) == 2
        x = self.token_emb(masked_ids)
        context_token = self.context_embed(context.type_as(x))
        x = torch.cat((context_token, x), dim=1)

        layer=0
        
        if self.skip:
            skips = []
        for blk in self.in_blocks:
            # 将adapter放在attention之后
            x = blk(x,adapter=self.adapter if use_adapter else None,layer=layer)
            if self.skip:
                skips.append(x)# type: ignore
            layer+=1
            
        x = self.mid_block(x)

        for blk in self.out_blocks:
            if self.skip:
                x = blk(x, skips.pop(),adapter = self.adapter if use_adapter else None,layer=layer)# type: ignore
            else:
                x = blk(x,adapter = self.adapter if use_adapter else None,layer=layer)

        x = self.norm(x)

        word_embeddings = self.token_emb.word_embeddings.weight.data.detach()
        x = self.mlm_layer(x, word_embeddings)
        x = x[:, self.extras:, :self.codebook_size]
        return x
