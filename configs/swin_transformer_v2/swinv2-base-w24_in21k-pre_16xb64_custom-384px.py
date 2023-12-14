# https://mmpretrain.readthedocs.io/en/latest/notes/pretrain_custom_dataset.html
# https://mmpretrain.readthedocs.io/en/latest/user_guides/config.html

_base_ = [ ## This config file will inherit all config files in `_base_`.
    '../_base_/models/swin_transformer_v2/base_384.py', # model settings
    '../_base_/datasets/imagenet_bs64_swin_384.py', # data settings
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',    # schedule settings
    '../_base_/default_runtime.py'  # runtime settings
]

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
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


model = dict(
    type='ImageClassifier',
    backbone=dict(
        img_size=384,
        window_size=[24, 24, 24, 12],
        drop_path_rate=0.2,
        pretrained_window_sizes=[12, 12, 12, 6],
        pad_small_map=True)) #newly added line
