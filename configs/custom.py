import ml_collections


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
    config.data_path="data/one_style.json"
    config.resume_root="assets/ckpts/cc3m-285000.ckpt"
    config.adapter_path=None
    config.sample_interval=True
    config.train = d(
        n_steps=1000,
        batch_size=8,
        log_interval=20,
        eval_interval=100,
        save_interval=100,
        fid_interval=20000,
        num_workers=8,
        resampled=False,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0003,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=-1, # 5000
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
        use_checkpoint=False,
        skip=True,
        d_prj=32,# Stage I: 32; Stage II: TODO 
        is_shared=False, # Stage I: False; Stage II: False
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
