num_classes = 2

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained="/home/lmf/mmsegmentation/weights/swin_base_patch4_window7_224_22k_20220317-4f79f7c0.pth",  # 将被加载的ImageNet预训练主干网络
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True,
        patch_size=4,
        mlp_ratio=4,
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg
        ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        # loss_decode=[
        #     dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.75),
        #     dict(type='DiceLoss', loss_weight=0.25)
        #     ]
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_weight=1.0),
            dict(type='LovaszLoss', loss_weight=1.0, per_image=False,reduction = 'none')
            ],
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)
        ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_weight=1.0),
            dict(type='LovaszLoss', loss_weight=1.0, per_image=False,reduction = 'none')
            ],
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)
        ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict())

# dataset settings
dataset_type = 'CustomDataset'
data_root = 'data/tamper/'
classes = ["0", "1"]
palette = [[0,0,0], [255,255,255]]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
size = 256
crop_size = 256
ratio = 2.0
model['test_cfg'] = dict(mode='slide', stride = (size, size), crop_size = (size, size))
albu_train_transforms = [
    dict(type='ColorJitter', p=0.5),
    # dict(type='GaussianBlur', p=0.5),
    # dict(type='JpegCompression', p=0.5, quality_lower = 75),
    # dict(type='Affine', rotate=5, shear=5, p=0.5),
    #dict(type='RandomResizedCrop', always_apply = True, height = crop_size, width = crop_size, scale = (0.9, 1.1), ratio = (1.0, 1.0), p=0.5),
    # dict(type='ToGray', p=0.5),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Albu', transforms=albu_train_transforms),
    dict(type='Resize', img_scale=(size, size)), #变化图象
    dict(type='RandomCrop', crop_size=(crop_size, crop_size), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate90', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(size, size), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=None,
        img_ratios=[ratio],
        flip=False,
        flip_direction=['horizontal', 'vertical'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(size, size), pad_val=0, seg_pad_val=255), #填充当前图像到指定大小
            dict(type='ImageToTensor', keys=['img']),#将图像转换为张量
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=12,#单个gpu的batch size
    workers_per_gpu=4,#单个gpu分配的数据加载线程数
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train_last/img',
        ann_dir='train_last/mask',
        img_suffix=".jpg",
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        #use_mosaic=False,
        #mosaic_prob=0.5,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train_last2/img',
        ann_dir='train_last2/msk',
        # img_dir='val_text/img_text',
        # ann_dir='val_text/msk_text',
        img_suffix=".jpg",
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        img_dir='test/imgs',
        #ann_dir='test/msk',
        img_suffix=".jpg",
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline))

log_config = dict(  #注册日志钩的配置文件
    interval=50, #打印日志的间隔
    hooks=[ #训练期间执行的钩子
        dict(type='TextLoggerHook'),    
        #dict(type='CustomizedTextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
#load_from = "./work_dirs/tamper/swin_b_team_30k/iter_30000.pth"
# load_from = '/home/lmf/mmsegmentation/work_dirs/tamper/swin_b_team_ft2_12k/epoch_70.pth'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

nx = 12
total_epochs = int(round(12 * nx))
optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# learning policy
lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# runtime settings

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs) #使用的runner类别，（epoch或iter）
checkpoint_config = dict(by_epoch=True, interval=5, save_optimizer=False,save_last=True,max_keep_ckpts=5) #检查点配置文件
evaluation = dict(by_epoch=True, interval=5, metric=['mIoU', 'mFscore'], pre_eval=True)
fp16 = dict(loss_scale=512.0)


work_dir = f'./work_dirs/tamper/swin_b_team_ft2_{nx}k'
