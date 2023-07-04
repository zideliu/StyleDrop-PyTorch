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

    config.train = d(
        n_steps=99999999,
        batch_size=2048,
        log_interval=10,
        eval_interval=5000,
        save_interval=5000,
        fid_interval=50000,
    )

    config.eval = d(
        n_samples=10000,
        sample_steps=12,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0004,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    config.nnet = d(
        name='uvit_vq',
        img_size=16,
        codebook_size=1024,
        in_chans=256,
        patch_size=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        num_classes=1001,
        use_checkpoint=False,
        skip=True,
    )

    config.muse = d(
        ignore_ind=-1,
        smoothing=0.1,
        gen_temp=4.5
    )

    config.dataset = d(
        name='imagenet256_features',
        path='assets/datasets/imagenet256_vq_features/vq-f16-jax',
        cfg=True,
        p_uncond=0.15,
    )

    config.sample = d(
        sample_steps=12,
        n_samples=50000,
        mini_batch_size=50,
        cfg=True,
        linear_inc_scale=True,
        scale=3.,
        path=''
    )

    return config
