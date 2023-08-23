import os
import gradio as gr
import open_clip
import torch
import taming.models.vqgan
import ml_collections
import einops
import random
# Model
from libs.muse import MUSE
import utils
import numpy as np
from glob import glob
empty_context = np.load("assets/contexts/empty_context.npy")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)

def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 1234
    config.z_shape = (8, 16, 16)

    config.autoencoder = d(
        config_file='vq-f16-jax.yaml',
    )
    config.resume_root="assets/ckpts/cc3m-285000.ckpt"
    config.adapter_path=None
    config.optimizer = d(
            name='adamw',
            lr=0.0002,
            weight_decay=0.03,
            betas=(0.99, 0.99),
    )
    config.lr_scheduler = d(
            name='customized',
            warmup_steps=5000
    )
    config.nnet = d(
        name='uvit_t2i_vq',
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
        is_shared=False
    )
    config.muse = d(
        ignore_ind=-1,
        smoothing=0.1,
        gen_temp=4.5
    )
    config.sample = d(
        sample_steps=36,
        n_samples=50,
        mini_batch_size=8,
        cfg=True,
        linear_inc_scale=True,
        scale=10.,
        path='',
        lambdaA=2.0, # Stage I: 2.0; Stage II: TODO
        lambdaB=5.0, # Stage I: 5.0; Stage II: TODO
    )
    return config

def cfg_nnet(x, context, scale=None,lambdaA=None,lambdaB=None):
    _cond = nnet_ema(x, context=context)
    _cond_w_adapter = nnet_ema(x,context=context,use_adapter=True)
    _empty_context = torch.tensor(empty_context, device=device)
    _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
    _uncond = nnet_ema(x, context=_empty_context)
    res = _cond + scale * (_cond - _uncond)
    if lambdaA is not None:
        res = _cond_w_adapter + lambdaA*(_cond_w_adapter - _cond) + lambdaB*(_cond - _uncond)
    return res

def unprocess(x):
    x.clamp_(0., 1.)
    return x

config = get_config()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

style_adapters = glob("style_adapter/*/")

style_adapters = [os.path.basename(os.path.dirname(x)) for x in style_adapters]

# Load open_clip and vq model
prompt_model,_,_ = open_clip.create_model_and_transforms('ViT-bigG-14', 'laion2b_s39b_b160k')
prompt_model = prompt_model.to(device)
prompt_model.eval()
tokenizer = open_clip.get_tokenizer('ViT-bigG-14')

vq_model = taming.models.vqgan.get_model('vq-f16-jax.yaml')
vq_model.eval()
vq_model.requires_grad_(False)
vq_model.to(device)

## config


muse = MUSE(codebook_size=vq_model.n_embed, device=device, **config.muse)

train_state = utils.initialize_train_state(config, device)
train_state.resume(ckpt_root=config.resume_root)
nnet_ema = train_state.nnet_ema
nnet_ema.eval()
nnet_ema.requires_grad_(False)
nnet_ema.to(device)

style_ref = {
    "None": None,
    **{x: os.path.join("style_adapter", x, "adapter.pth") for x in style_adapters}
}
# style_postfix = {
#     "None": "",
#     **{x: f" in {x.replace('_', ' ')} style" for x in style_adapters}
# }
style_postfix ={
    "None":"",
    "0102":" in watercolor painting style",
    "0103":" in watercolor painting style",
    "0106":" in line drawing style",
    "0108":" in oil painting style",
    "0301":" in 3d rendering style",
    "0305":" in kid crayon drawing style",
}
def decode(_batch):
    return vq_model.decode_code(_batch)

def process(prompt,num_samples,lambdaA,lambdaB,style,seed,sample_steps,image=None):
    config.sample.lambdaA = lambdaA
    config.sample.lambdaB = lambdaB
    config.sample.sample_steps = sample_steps
    print(style)
    adapter_path = style_ref[style]
    adapter_postfix = style_postfix[style]
    print(f"load adapter path: {adapter_path}")
    if adapter_path is not None:
        nnet_ema.adapter.load_state_dict(torch.load(adapter_path))
    else:
        config.sample.lambdaA=None
        config.sample.lambdaB=None
    print("load adapter Done!")
    # Encode prompt
    prompt = prompt+adapter_postfix
    text_tokens = tokenizer(prompt).to(device)
    text_embedding = prompt_model.encode_text(text_tokens)
    text_embedding = text_embedding.repeat(num_samples, 1, 1) # B 77 1280
    print(text_embedding.shape)
   
    print(f"lambdaA: {lambdaA}, lambdaB: {lambdaB}, sample_steps: {sample_steps}")
    if seed==-1:
        seed = random.randint(0,65535)
    config.seed = seed
    print(f"seed: {seed}")
    set_seed(config.seed)
    res = muse.generate(config,num_samples,cfg_nnet,decode,is_eval=True,context=text_embedding)
    print(res.shape)
    res = (res*255+0.5).clamp_(0,255).permute(0,2,3,1).to('cpu',torch.uint8).numpy()
    im = [res[i] for i in range(num_samples)]
    return im
    
    
    
block = gr.Blocks()
with block:
    with gr.Row():
        gr.Markdown("## StyleDrop based on Muse (Inference Only) ")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
            seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=1234)
            style = gr.Radio(choices=style_adapters+["None"], type="value",value="None",label="Style")

            with gr.Accordion("Advanced options",open=False):
                lambdaA = gr.Slider(label="lambdaA", minimum=0.0, maximum=5.0, value=2.0, step=0.01)
                lambdaB = gr.Slider(label="lambdaB", minimum=0.0, maximum=10.0, value=5.0, step=0.01)
                sample_steps = gr.Slider(label="Sample steps", minimum=1, maximum=50, value=36, step=1)
                image=gr.Image(value=None)
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(columns=2, height='auto')
            
    with gr.Row():
        examples = [
            [   "data/image_01_03.jpg",
                "A banana on the table",
                1,2.0,5.0,"0103",1234,36,
            ],
            [
                "data/image_01_02.jpg",
                "A cow",
                1,2.0,5.0,"0102",1234,36
            ],
            [
                "data/image_01_06.jpg",
                "A portrait of tabby cat",
                1,2.0,5.0,"0106",1234,36,
            ],
            [   
                "data/image_01_08.jpg",
                "A church in the field",
                1,2.0,5.0,"0108",1234,36,
            ],
            [
                "data/image_03_05.jpg",
                "A Christmas tree",
                1,2.0,5.0,"0305",1234,36,
            ]
            
        ]
        gr.Examples(examples=examples,
                        fn=process,
                        inputs=[
                            image,
                            prompt,
                            num_samples,lambdaA,lambdaB,style,seed,sample_steps,
                        ],
                        outputs=result_gallery,
                        cache_examples=os.getenv('SYSTEM') == 'spaces'
                        )
    ips = [prompt,num_samples,lambdaA,lambdaB,style,seed,sample_steps,image]
    run_button.click(
        fn=process,
        inputs=ips,
        outputs=[result_gallery]
    )
block.launch(share=True,show_error=True)

