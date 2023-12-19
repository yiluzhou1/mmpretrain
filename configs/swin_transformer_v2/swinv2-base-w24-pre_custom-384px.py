# use the SwinTransformerV2 as the backbone for the MAE (Masked AutoEncoder) algorithm
# https://mmpretrain.readthedocs.io/en/latest/notes/pretrain_custom_dataset.html
# https://mmpretrain.readthedocs.io/en/latest/user_guides/config.html
# https://mmpretrain.readthedocs.io/en/latest/api/models.html

_base_ = [
    '../_base_/models/mae_vit-base-p16.py',  # MAE model settings
    '../_base_/datasets/imagenet_bs512_mae.py',  # data settings
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',  # schedule settings
    '../_base_/default_runtime.py'  # runtime settings
]

# >>>>>>>>>>>>>>> Override model settings here >>>>>>>>>>>>>>>>>>>


model = dict(
    type='MAE',
    backbone=dict(
        type='SwinTransformerV2',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False
    ),
    neck=dict(
        type='MLPNeck',
        in_channels=768,
        hidden_channels=2048,
        out_channels=2048,
        num_layers=2
    ),
    head=dict(
        type='ContrastiveHead',
        dimension=2048,
        contrastive_temperature=0.1
    )
)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>>>>>>>>>>>>>> Override dataset settings here >>>>>>>>>>>>>>>>>>>
# img_norm_cfg = dict(
#     mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
#     std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
#     to_rgb=True)

# data = dict(
#     train=dict(
#         type='CustomDataset',
#         data_root='data/custom_dataset/',
#         ann_file='',  # We assume you are using the sub-folder format without ann_file
#         data_prefix='',  # The `data_root` is the data_prefix directly.
#         with_label=False,
#         batch_size=64,
#         num_workers=4,
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(type='RandomResizedCrop', size=224),
#             dict(type='RandomFlip', flip_prob=0.5),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='ToTensor', keys=['gt_label']),
#             dict(type='Collect', keys=['img', 'gt_label'])
#         ]
#     )
# )

# >>>>>>>>>>>>>>> Override dataset settings here >>>>>>>>>>>>>>>>>>>
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    dataset=dict(
        type='CustomDataset',
        data_root='data/custom_dataset/',
        ann_file='',       # We assume you are using the sub-folder format without ann_file
        data_prefix='',    # The `data_root` is the data_prefix directly.
        with_label=False,
    )
)

val_cfg = None
test_cfg = None
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# model = dict(
#     type='ImageClassifier',
#     backbone=dict(
#         img_size=384,
#         window_size=[24, 24, 24, 12],
#         drop_path_rate=0.2,
#         pretrained_window_sizes=[12, 12, 12, 6],
#         pad_small_map=True)) #newly added line

"""
python tools/train.py configs/swin_transformer_v2/swinv2-base-w24-pre_custom-384px.py

"""