num_classes = 2
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='./weights/convnext_large_22k_224.pth',
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_weight=1.0),
            dict(type='LovaszLoss', loss_weight=1.0, per_image=True)
        ],
        sampler=dict(type='OHEMPixelSampler', thresh=0.6, min_kept=100000)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_weight=1.0),
            dict(type='LovaszLoss', loss_weight=1.0, per_image=True)
        ],
        sampler=dict(type='OHEMPixelSampler', thresh=0.6, min_kept=100000)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=(256, 256), crop_size=(256, 256)))
dataset_type = 'CustomDataset'
data_root = 'data/tamper/'
classes = ['0', '1']
palette = [[0, 0, 0], [255, 255, 255]]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
size = 256
crop_size = 256
ratio = 1.0
albu_train_transforms = [dict(type='ColorJitter', p=0.5)]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomCrop', cat_max_ratio=0.9, crop_size=(256, 256)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate90', prob=0.5),
    dict(type='Albu', transforms=[dict(type='ColorJitter', p=0.5)]),
    dict(type='Resize', img_scale=(256, 256)),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=None,
        img_ratios=[1.0],
        flip=False,
        flip_direction=['horizontal', 'vertical'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=2,
    train=dict(
        type='CustomDataset',
        data_root='data/tamper/',
        img_dir='train/img',
        ann_dir='train/msk',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=['0', '1'],
        palette=[[0, 0, 0], [255, 255, 255]],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='RandomCrop', cat_max_ratio=0.9, crop_size=(256, 256)),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='RandomFlip', prob=0.5, direction='vertical'),
            dict(type='RandomRotate90', prob=0.5),
            dict(type='Albu', transforms=[dict(type='ColorJitter', p=0.5)]),
            dict(type='Resize', img_scale=(256, 256)),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='CustomDataset',
        data_root='data/tamper/',
        img_dir='train2/img',
        ann_dir='train2/msk',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=['0', '1'],
        palette=[[0, 0, 0], [255, 255, 255]],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=None,
                img_ratios=[1.0],
                flip=False,
                flip_direction=['horizontal', 'vertical'],
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CustomDataset',
        data_root='data/tamper/',
        test_mode=True,
        img_dir='test/imgs',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=['0', '1'],
        palette=[[0, 0, 0], [255, 255, 255]],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=None,
                img_ratios=[1.0],
                flip=False,
                flip_direction=['horizontal', 'vertical'],
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/lmf/mmsegmentation/work_dirs/tamper/convx_l_12x_dice_aug1_dec/epoch_144.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
nx = 12
total_epochs = 144
optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        decay_rate=0.99, decay_type='stage_wise', num_layers=12))
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.0)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=144)
checkpoint_config = dict(
    by_epoch=True,
    interval=10,
    save_optimizer=False,
    save_last=True,
    max_keep_ckpts=5)
evaluation = dict(
    by_epoch=True, interval=10, metric=['mIoU', 'mFscore'], pre_eval=True)
fp16 = dict(loss_scale=512.0)
work_dir = './work_dirs/tamper/NEWTRY_convx_l_12x_dice_aug1_dec'