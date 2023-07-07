# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md


import os
from typing import List
import open_clip
import torch
import ml_collections
import einops
import random
import numpy as np
from PIL import Image
from cog import BasePredictor, Input, Path

import taming.models.vqgan
from libs.muse import MUSE
import utils


empty_context = np.load("assets/contexts/empty_context.npy")

STYLES_CKPT = {
    "watercolor painting_1": "style_adapter/0102.pth",
    "watercolor painting_2": "style_adapter/0103.pth",
    "line drawing ": "style_adapter/0106.pth",
    "oil painting": "style_adapter/0108.pth",
    "3d rendering": "style_adapter/0301.pth",
    "kid crayon drawing": "style_adapter/0305.pth",
}


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.config = get_config()
        self.device = torch.device("cuda:0")

        # Load open_clip and vq model
        self.prompt_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-bigG-14", "laion2b_s39b_b160k", cache_dir="assets/clip"
        )
        self.prompt_model = self.prompt_model.to(self.device)
        self.prompt_model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-bigG-14")

        self.vq_model = taming.models.vqgan.get_model("vq-f16-jax.yaml")
        self.vq_model.eval()
        self.vq_model.requires_grad_(False)
        self.vq_model.to(self.device)

        ## config

        self.muse = MUSE(
            codebook_size=self.vq_model.n_embed, device=self.device, **self.config.muse
        )

        train_state = utils.initialize_train_state(self.config, self.device)
        train_state.resume(ckpt_root=self.config.resume_root)
        self.nnet_ema = train_state.nnet_ema
        self.nnet_ema.eval()
        self.nnet_ema.requires_grad_(False)
        self.nnet_ema.to(self.device)

    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
        style_adapter: str = Input(
            description="Choose a style adapter. Note that only the pretrained styles here https://huggingface.co/zideliu/StyleDrop/tree/main are available.",
            choices=list(STYLES_CKPT.keys()),
            default="oil painting",
        ),
        num_samples: int = Input(
            description="Number of images to output.",
            ge=1,
            le=12,
            default=1,
        ),
        sample_steps: int = Input(
            description="Set sampling step",
            ge=1,
            le=50,
            default=36,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        self.config.seed = seed
        set_seed(seed)
        self.config.sample.sample_steps = sample_steps

        self.nnet_ema.adapter.load_state_dict(torch.load(STYLES_CKPT[style_adapter]))

        # Encode prompt
        prompt = f"{prompt} in {style_adapter.split('_')[0]} style"
        print(f"prompt: {prompt}")
        text_tokens = self.tokenizer(prompt).to(self.device)
        text_embedding = self.prompt_model.encode_text(text_tokens)
        text_embedding = text_embedding.repeat(num_samples, 1, 1)  # B 77 1280
        print(text_embedding.shape)

        def cfg_nnet(x, context, scale=None, lambdaA=None, lambdaB=None):
            _cond = self.nnet_ema(x, context=context)
            _cond_w_adapter = self.nnet_ema(x, context=context, use_adapter=True)
            _empty_context = torch.tensor(empty_context, device=self.device)
            _empty_context = einops.repeat(_empty_context, "L D -> B L D", B=x.size(0))
            _uncond = self.nnet_ema(x, context=_empty_context)
            res = _cond + scale * (_cond - _uncond)
            if lambdaA is not None:
                res = (
                    _cond_w_adapter
                    + lambdaA * (_cond_w_adapter - _cond)
                    + lambdaB * (_cond - _uncond)
                )
            return res

        res = self.muse.generate(
            self.config,
            num_samples,
            cfg_nnet,
            self.vq_model.decode_code,
            is_eval=True,
            context=text_embedding,
        )

        res = (
            (res * 255 + 0.5)
            .clamp_(0, 255)
            .permute(0, 2, 3, 1)
            .to("cpu", torch.uint8)
            .numpy()
        )

        out_images = [res[i] for i in range(num_samples)]
        output = []
        for i, img in enumerate(out_images):
            out = f"/tmp/out_{i}.png"
            img = Image.fromarray(img)
            img.save(out)
            output.append(Path(out))
        return output


def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 1234
    config.z_shape = (8, 16, 16)

    def get_dict(**kwargs):
        """Helper of creating a config dict."""
        return ml_collections.ConfigDict(initial_dictionary=kwargs)

    config.autoencoder = get_dict(config_file="vq-f16-jax.yaml")
    config.resume_root = "assets/ckpts/cc3m-285000.ckpt"
    config.adapter_path = None
    config.optimizer = get_dict(
        name="adamw",
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )
    config.lr_scheduler = get_dict(name="customized", warmup_steps=5000)
    config.nnet = get_dict(
        name="uvit_t2i_vq",
        img_size=16,
        codebook_size=1024,
        in_chans=4,
        embed_dim=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        clip_dim=1280,
        num_clip_token=77,
        use_checkpoint=True,
        skip=True,
        d_prj=32,
        is_shared=False,
    )
    config.muse = get_dict(ignore_ind=-1, smoothing=0.1, gen_temp=4.5)
    config.sample = get_dict(
        sample_steps=36,
        n_samples=50,
        mini_batch_size=8,
        cfg=True,
        linear_inc_scale=True,
        scale=10.0,
        path="",
        lambdaA=2.0,  # Stage I: 2.0; Stage II: TODO
        lambdaB=5.0,  # Stage I: 5.0; Stage II: TODO
    )
    return config


def unprocess(x):
    x.clamp_(0.0, 1.0)
    return x


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
