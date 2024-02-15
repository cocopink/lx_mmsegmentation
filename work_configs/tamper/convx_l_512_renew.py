num_classes = 2

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)  #   分割框架通常使用SyncBN
model = dict(
    type='EncoderDecoder',  # 分割器（segmentor）的名字
    pretrained="./weights/convnext_large_22k_224.pth",  # 将被加载的ImageNet预训练主干网络
    backbone=dict(
        type='ConvNeXt',    # 主干网络的类别
        in_chans=3,
        depths=[3, 3, 27, 3], 
        dims=[192, 384, 768, 1536], 
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
    ),
    decode_head=dict(
        type='UPerHead', # 解码头（decode head）的类别
        in_channels=[192, 384, 768, 1536],  #解码头的输入通道数
        in_index=[0, 1, 2, 3],  #解码头的特征图索引
        pool_scales=(1, 2, 3, 6),   #平均池化的规模（scales）
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,    #分割前景的类别
        norm_cfg=norm_cfg,          #归一化层的配置项
        align_corners=False,        #解码里调整大小（resize）的参数
        # loss_decode = [
        #     dict(type = 'CrossEntropyLoss', loss_weight=1.0,class_weight=[1,8]),
        #     dict(type='DiceLoss', loss_weight=1.0)
        # ],
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_weight=1.0),
            dict(type='LovaszLoss', loss_weight=1.0, per_image=False,reduction = 'none')
            ],
        sampler=dict(type='OHEMPixelSampler', thresh=0.6, min_kept=100000)
        ),
    auxiliary_head=dict(
        type='FCNHead', # 辅助头（auxiliary head）的种类
        in_channels=768,    # 辅助头的输入通道数
        in_index=2, #被选择的特征图索引
        channels=256,   #辅助头中间态的通道数
        num_convs=1,    #FCNHead里卷积的数目，辅助头里通常为1
        concat_input=False, #在分类层（classification layer）之前是否连接输入和卷积的输出
        dropout_ratio=0.1,  #进入最后分类层之前的dropout比例
        num_classes=num_classes,    #
        norm_cfg=norm_cfg,  #归一化层的配置项
        align_corners=False,
        # loss_decode = [
        #     dict(type = 'CrossEntropyLoss', loss_weight=1.0,class_weight=[1,8]),
        #     dict(type='DiceLoss', loss_weight=1.0)
        # ],
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_weight=1.0),
            dict(type='LovaszLoss', loss_weight=1.0, per_image=False,reduction = 'none')
            ],
        sampler=dict(type='OHEMPixelSampler', thresh=0.6, min_kept=100000)
        ),
    # model training and testing settings
    train_cfg=dict(),   #当前近视一个占位符
    test_cfg=dict())

# dataset settings
dataset_type = 'CustomDataset'  #数据集类型，用来定义数据集
data_root = 'data/tamper/'
classes = ["0", "1"]
palette = [[0,0,0], [255,255,255]]
img_norm_cfg = dict(    #图像归一化配置，用来归一化输入的图像
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)   #预训练里用于预训练主干网络模型的平均值和标准差以及图像的通道顺序
size = 512
size_x = 512
size_y = 512
crop_size = 512 #训练时的裁剪大小
ratio = size / crop_size
model['test_cfg'] = dict(mode='slide', stride = (size, size), crop_size = (size_x, size_y))
# crop_size = (256, 256)
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
    dict(type='LoadAnnotations'),   #加载图像和其注释信息
    dict(type='RandomCrop', cat_max_ratio = 0.9, crop_size = (size_x, size_y)), #随机裁剪当前图像和其注释大小的数据增广的流程
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),  #翻转
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate90', prob=0.5),
    dict(type='Albu', transforms=albu_train_transforms),
    dict(type='Resize', img_scale=(size_x, size_y)), #变化图象
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(size_x, size_y), pad_val=0, seg_pad_val=255), #填充当前图像到指定大小
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
            dict(type='Pad', size=(size_x, size_y), pad_val=0, seg_pad_val=255), #填充当前图像到指定大小
            dict(type='ImageToTensor', keys=['img']),#将图像转换为张量
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,#单个gpu的batch size
    workers_per_gpu=2,#单个gpu分配的数据加载线程数
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/img',
        ann_dir='train/msk',
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
        img_dir='train2/img',
        ann_dir='train2/msk',
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


# yapf:disable
log_config = dict(  #注册日志钩的配置文件
    interval=50, #打印日志的间隔
    hooks=[ #训练期间执行的钩子
        dict(type='TextLoggerHook'),    
        #dict(type='CustomizedTextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl') #用于设置分布式训练的参数，端口也同样可被设置
log_level = 'INFO'  #日志的级别
# load_from = "/home/lmf/mmsegmentation/work_dirs/tamper/convx_l_12x_dice_aug1_dec/epoch_144.pth"
load_from = None # "./work_dirs/tamper/convx_t_8x/epoch_96.pth" 从一个给定路径里加载模型作为预训练模型，并不会消耗训练时间
resume_from = None # 从给定路径里恢复检查点(checkpoints)，训练模式将从检查点保存的轮次开始恢复训练。
workflow = [('train', 1)]# runner 的工作流程。 [('train', 1)] 意思是只有一个工作流程而且工作流程 'train' 仅执行一次。根据 `runner.max_iters` 工作流程训练模型的迭代轮数为40000次。
cudnn_benchmark = True# 是否是使用 cudnn_benchmark 去加速，它对于固定输入大小的可以提高训练速度。

nx = 12
total_epochs = int(round(12 * nx))
# optimizer 优化器种类 学习率 动量 衰减权重
optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', type='AdamW', 
                 lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 0.99,
                                'decay_type': 'stage_wise',
                                'num_layers': 12})
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
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

work_dir = f'./work_dirs/tamper/512_nex_convx_l_{nx}x_dice_aug1_dec'