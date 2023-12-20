_base_ = [
    '../_base_/models/mae_vit-base-p16.py',
    '../_base_/datasets/imagenet_bs512_mae.py',
    '../_base_/default_runtime.py',
]

# >>>>>>>>>>>>>>> Override dataset settings here >>>>>>>>>>>>>>>>>>>
train_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root='data/custom_dataset/',
        ann_file='',       # We assume you are using the sub-folder format without ann_file
        data_prefix='',    # The `data_root` is the data_prefix directly.
        with_label=False,
    )
)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# optimizer wrapper
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=1.5e-4 * 4096 / 256,
        betas=(0.9, 0.95),
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=260,
        by_epoch=True,
        begin=40,
        end=300,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300)
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

randomness = dict(seed=0, diff_rank_seed=True)

# auto resume
resume = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=4096)
"""

python tools/train.py configs/mae/mae_vit-base-p16_8xb512-amp-coslr-300e_custom.py

https://mmpretrain.readthedocs.io/en/latest/notes/pretrain_custom_dataset.html
https://mmpretrain.readthedocs.io/en/latest/user_guides/config.html
https://mmpretrain.readthedocs.io/en/latest/api/models.html
"""