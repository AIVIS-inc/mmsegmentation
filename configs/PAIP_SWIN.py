_base_ = [
    './swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_'
    'pretrain_224x224_1K.py'
]
model = dict(
    pretrained=\
    'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth', # noqa
    backbone=dict(
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32]),
    decode_head=dict(
        in_channels=[128, 256, 512, 1024],
        dropout_ratio=0.5,
        num_classes=2),
    auxiliary_head=dict(
        in_channels=512,
        dropout_ratio=0.5, 
        num_classes=2))
# optimizer
#optimizer = dict(type='Adam', lr=0.001, eps=1e-08, weight_decay=0.0005)
#optimizer_config = dict()
# learning policy
#lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict( type='Normalize',  # Normalization config, the values are from img_norm_cfg
          mean=[123.675, 116.28, 103.53],
          std=[58.395, 57.12, 57.375],
          to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
dict(type='LoadImageFromFile'),  # First pipeline to load images from file path
dict(
type='MultiScaleFlipAug',  # An encapsulation that encapsulates the test time augmentations
img_scale=(512, 512),  # Decides the largest scale for testing, used for the Resize pipeline
flip=False,  # Whether to flip images during testing
transforms=[
dict(type='RandomFlip'),  # Thought RandomFlip is added in pipeline, it is not used when flip=False
dict(
type='Normalize',  # Normalization config, the values are from img_norm_cfg
mean=[123.675, 116.28, 103.53],
std=[58.395, 57.12, 57.375],
to_rgb=True),
dict(type='ImageToTensor', keys=['img']),
dict(type='Collect', keys=['img'])]
)
]
#train_cfg = dict()  # train_cfg is just a place holder for now.
#test_cfg = dict(mode='whole')
data = dict(
    samples_per_gpu=6,  # Batch size of a single GPU
    workers_per_gpu=4,  # Worker to pre-fetch data for each single GPU
    train=dict(  # Train dataset config
        type='DRIVEDataset',  # Type of dataset, refer to mmseg/datasets/ for details.
        data_root='/home/ubuntu/DATA_PAIP/Nerve_ALL',  # The root of dataset.
        img_dir='train/IMG',  # The image directory of dataset.
        ann_dir='train/GT',  # The annotation directory of dataset.
        pipeline=train_pipeline),
    val=dict(  # Validation dataset config
        type='DRIVEDataset',
        data_root='/home/ubuntu/DATA_PAIP/Nerve_ALL',  # The root of dataset.
        img_dir='train/IMG',  # The image directory of dataset.
        ann_dir='train/GT',  # The annotation directory of dataset.
        pipeline=test_pipeline)
        )
log_config = dict(  # config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[
        # dict(type='TensorboardLoggerHook')  # The Tensorboard logger is also supported
        dict(type='TextLoggerHook', by_epoch=False)
    ])

log_level = 'INFO'  # The level of logging.
load_from = None  # load models as a pre-trained model from a given path. This will not resume training.
resume_from = None  
workflow = [('train', 1)] 
cudnn_benchmark = True 

runner = dict(
    type='IterBasedRunner', # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
    max_iters=10000) # Total number of iterations. For EpochBasedRunner use `max_epochs`
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    by_epoch=False,  # Whethe count by epoch or not.
    interval=2000)  # The save interval.
evaluation = dict(  # The config to build the evaluation hook. Please refer to mmseg/core/evaulation/eval_hook.py for details.
    interval=1000,  # The interval of evaluation.
    metric='mIoU')  # The evaluation metric.