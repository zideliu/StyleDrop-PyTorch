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
        n_steps=999999999,
        batch_size=2048,
        log_interval=10,
        eval_interval=5000,
        save_interval=5000,
        fid_interval=50000,
        num_workers=8,
        resampled=False,
    )

    config.eval = d(
        n_samples=10000,
        sample_steps=18,
    )

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
    )

    config.muse = d(
        ignore_ind=-1,
        smoothing=0.1,
        gen_temp=4.5
    )

    config.dataset = d(
        name='cc3m_web',
        cfg=True,
        p_uncond=0.15,
    )

    config.wds = d(
        train_data='assets/datasets/cc3m/vq_f16_jax_clipG_cc3m_train_emb/{00000..03044}.tar',
        val_data='assets/datasets/cc3m/vq_f16_jax_clipG_cc3m_val_emb/{00000..00012}.tar',
        ctx_path='assets/contexts',
        dist_eval=True,
    )

    config.sample = d(
        sample_steps=18,
        n_samples=30000,
        mini_batch_size=2,
        cfg=True,
        linear_inc_scale=True,
        scale=10.,
        path='',
    )

    return config
