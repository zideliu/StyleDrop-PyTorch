
import os
import builtins
import datetime
import time
from argparse import Namespace

import accelerate
import einops
import ml_collections
import torch
from datasets import get_dataset
from loguru import logger
from torch import multiprocessing as mp
from torch.utils._pytree import tree_map
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import taming.models.vqgan
import wandb
import torch.utils.data
from libs.muse import MUSE
from tools.fid_score import calculate_fid_given_paths
from custom.custom_dataset import test_custom_dataset,train_custom_dataset,Discriptor
import utils

import open_clip
import taming.models.vqgan
logging = logger

torch.multiprocessing.set_sharing_strategy('file_system')  # todo


def convert_model_dtype(models, dtype):
    logging.info(f'Converting model to {dtype}')
    if not isinstance(models, (list, tuple)):
        models = [models]
    for model in models:
        if model is None:
            continue
        if dtype == torch.float16:
            model.half()
        elif dtype == torch.bfloat16:
            model.bfloat16()


def LSimple(x0, nnet, schedule, **kwargs):
    labels, masked_ids = schedule.sample(x0)
    logits = nnet(masked_ids, **kwargs, use_adapter=True)
    # b (h w) c, b (h w)
    loss = schedule.loss(logits, labels)
    return loss


def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.ConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes
    
    # Load open_clip and vq model
    prompt_model,_,_ = open_clip.create_model_and_transforms('ViT-bigG-14', 'laion2b_s39b_b160k')
    prompt_model = prompt_model.to(device)
    prompt_model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
    
    vq_model = taming.models.vqgan.get_model('vq-f16-jax.yaml')
    vq_model.eval()
    vq_model.requires_grad_(False)
    vq_model.to(device)
    
    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logging.info(config)
        wandb.init(dir=os.path.abspath(config.workdir), project=f'cc3m', config=config.to_dict(),
                   job_type='train', mode='online', settings=wandb.Settings(start_method='fork'))
    else:
        logging.remove()
        logger.add(sys.stderr, level='ERROR')
        builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')

    ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(config.ckpt_root)))
    if not ckpts:
        resume_step = 0
    else:
        steps = map(lambda x: int(x.split(".")[0]), ckpts)
        resume_step = max(steps)

    logger.info(f'world size is {accelerator.num_processes}')



    dataset = train_custom_dataset(
        train_file=config.data_path,
    )
    test_dataset = test_custom_dataset(dataset.style)
    
    discriptor = Discriptor(dataset.style)
    
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.train.batch_size,
    )
    test_dataset_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=10,
    )

    prompt_loader = torch.utils.data.DataLoader(
        dataset = discriptor,
        batch_size = config.sample.mini_batch_size,
    )

    autoencoder = taming.models.vqgan.get_model(**config.autoencoder)
    autoencoder.to(device)

    train_state = utils.initialize_train_state(config, device)
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.resume_root,config.adapter_path)

    train_state.freeze()
    nnet, nnet_ema, optimizer = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer)
    

    
    @torch.cuda.amp.autocast()#type: ignore
    def encode(_batch):
        res = autoencoder.encode(_batch)[-1][-1].reshape(len(_batch), -1)
        return res

    @torch.cuda.amp.autocast()#type: ignore
    def decode(_batch):
        return autoencoder.decode_code(_batch)

    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                image, prompt = data
                prompt = list(prompt)
                text_tokens = tokenizer(prompt).to(device)
                text_embedding = prompt_model.encode_text(text_tokens).detach().cpu()
                
                image = image.to(device)
                image_embedding = vq_model(image)[-1][-1].detach().cpu()
                image_embedding = image_embedding.unsqueeze(dim=0)
                yield [image_embedding, text_embedding]

    data_generator = get_data_generator()

    def get_context_generator():
        while True:
            for data in test_dataset_loader:
                _, _context = data
                _context = list(_context)
                text_tokens = tokenizer(_context).to(device)
                _context = prompt_model.encode_text(text_tokens).detach().cpu()
                yield _context

    context_generator = get_context_generator()

    def get_eval_context():
        while True:
            for data in prompt_loader:
                prompt = list(data)
                text_tokens = tokenizer(prompt).to(device)
                embedding = prompt_model.encode_text(text_tokens).detach().cpu()
                yield prompt,embedding
    prompt_generator = get_eval_context()
    
    def get_constant_context():

        for data in test_dataset_loader:
            prompt,_context = data
            _context = list(_context)
            break
        text_tokens = tokenizer(_context).to(device)
        embedding = prompt_model.encode_text(text_tokens).detach().cpu()
        return prompt,embedding
    
    muse = MUSE(codebook_size=autoencoder.n_embed, device=device, **config.muse)

    def cfg_nnet(x, context, scale=None,lambdaA=None,lambdaB=None):
        _cond = nnet_ema(x, context=context)
        _cond_w_adapter = nnet_ema(x,context=context,use_adapter=True)
        
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet_ema(x, context=_empty_context)
        res = _cond + scale * (_cond - _uncond)
        if lambdaA is not None:
            res = _cond_w_adapter + lambdaA*(_cond_w_adapter - _cond) + lambdaB*(_cond - _uncond)
        return res

    def train_step(_batch):
        _metrics = dict()
        optimizer.zero_grad()
        _z, context = proc_batch_feat(_batch)
        loss = LSimple(_z, nnet, muse, context=context)  # currently only support the extracted feature version
        metric_logger.update(loss=accelerator.gather(loss.detach()).mean())
        accelerator.backward(loss.mean())
        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        loss_scale, grad_norm = accelerator.scaler.get_scale(), utils.get_grad_norm_(nnet.parameters())  # type: ignore
        metric_logger.update(loss_scale=loss_scale)
        metric_logger.update(grad_norm=grad_norm)
        return dict(lr=train_state.optimizer.param_groups[0]['lr'],
                    **{k: v.value for k, v in metric_logger.meters.items()})

    def proc_batch_feat(_batch):
        _z = _batch[0].reshape(-1, 256)
        context = _batch[1].reshape(_z.shape[0], 77, -1)
        assert context.shape[-1] == config.nnet.clip_dim # type: ignore
        return _z, context

    def eval_step(n_samples, sample_steps):
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}'
                     f'mini_batch_size={config.sample.mini_batch_size}') # type: ignore

        def sample_fn(_n_samples):
            # _context = next(context_generator)
            prompt, _context = next(prompt_generator)
            _context = _context.to(device).reshape(-1, 77, config.nnet.clip_dim)
            kwargs = dict(context=_context)
            return muse.generate(config, _context.shape[0], cfg_nnet, decode,is_eval=True, **kwargs)

        if accelerator.is_main_process:
            path = f'{config.workdir}/eval_samples/{train_state.step}_{datetime.datetime.now().strftime("%m%d_%H%M%S")}'
            logging.info(f'Path for Eval images: {path}')
        else:
            path = None

        utils.sample2dir(accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn,  # type: ignore
                         dataset.unpreprocess)

        return 0

    if eval_ckpt_path := os.getenv('EVAL_CKPT', ''):
        adapter_path = os.getenv('ADAPTER',None)
        nnet.eval()
        train_state.resume(eval_ckpt_path,adapter_path)
        logging.info(f'Eval {train_state.step}...')
        eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps) # type: ignore
        return

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')
    step_fid = []
    metric_logger = utils.MetricLogger()
    cur_step = train_state.step
    while train_state.step < config.train.n_steps + cur_step:   # type: ignore
        nnet.train()
        data_time_start = time.time()
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        metric_logger.update(data_time=time.time() - data_time_start)
        metrics = train_step(batch)

        nnet.eval()

        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:   # type: ignore
            torch.cuda.empty_cache()
            logging.info(f'Save checkpoint {train_state.step}...')
            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'),adapter_only=True)
                if config.sample_interval:
                    eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps) # type: ignore
        accelerator.wait_for_everyone()

        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:   # type: ignore
            logger.info(f'step: {train_state.step} {metric_logger}')
            wandb.log(metrics, step=train_state.step)

        if train_state.step % config.train.eval_interval == 0:   # type: ignore
            torch.cuda.empty_cache()
            logging.info('Save a grid of images...')
            # contexts = torch.tensor(dataset.contexts, device=device)[: 2 * 5]
            prompt, contexts = get_constant_context()
            contexts = contexts.to(device)
            print(f"Eval prompt: {prompt[0]}")
            print(f"Shape of contexts :[{contexts.shape}]")
            samples = muse.generate(config, contexts.shape[0], cfg_nnet, decode, context=contexts)
            samples = make_grid(dataset.unpreprocess(samples), contexts.shape[0])
            save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}_{accelerator.process_index}.png'))
            if accelerator.is_main_process:
                wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        if train_state.step % config.train.fid_interval == 0 or train_state.step == config.train.n_steps:   # type: ignore
            torch.cuda.empty_cache()
            logging.info(f'Eval {train_state.step}...')
            fid = eval_step(n_samples=config.eval.n_samples,   # type: ignore
                            sample_steps=config.eval.sample_steps)  # calculate fid of the saved checkpoint   # type: ignore
            step_fid.append((train_state.step, fid))
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        
        
        
    del metrics


from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_bool("disable_val", False, 'help')


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem


def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.workdir = os.getenv('OUTPUT_DIR',
                               Path.home() / 'exp/default' / datetime.datetime.now().strftime("%m%d_%H%M%S"))
    config.disable_val = FLAGS.disable_val
    stage = "_I" if config.nnet.is_shared else "_II"
    # stage = "_II"
    config.ckpt_root = os.path.join(config.workdir, 'ckpts'+stage)
    
    # config.resume_root = "assets/ckpts/cc3m-285000.ckpt"
    config.sample_dir = os.path.join(config.workdir, 'samples'+stage)
    train(config)


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    app.run(main)
