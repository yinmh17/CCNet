model = dict(
    type='basenet',
    pretrained='',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        block_num=[3, 4, 23, 3],

    ),
    att=dict(
        with_att=False,
        type='glore',
        att_stage=[False,False,True,False],
        att_pos='after_add',
        att_location=[[],[],[5,11,17],[]],
    ),
    module=dict(
        type='nl_nowd',
        downsample=False,
        whiten_type=['ln_nostd'],
        weight_init_scale=1.0,
        with_unary=True,
        with_gc=False,
        with_nl=True,
        nowd=['nl'],
        use_out=False,
        out_bn=False,
    )
)
train_cfg = dict(
    batch_size=8,
    learning_rate=1e-2,
    momentum=0.9,
    num_steps=60000,
    power=0.9,
    random_seed=1234,
    restore_from='./dataset/resnet101-imagenet.pth',
    save_num_images=2,
    start_iters=0,
    save_from=59500,
    save_pred_every=100,
    snapshot_dir='snapshots/',
    weight_decay=0.0005
)
data_cfg = dict(
    data_dir='cityscapes',
    data_list='./dataset/list/cityscapes/train.lst',
    ignore_label=255,
    input_size='769,769',
    num_classes=19,
)
