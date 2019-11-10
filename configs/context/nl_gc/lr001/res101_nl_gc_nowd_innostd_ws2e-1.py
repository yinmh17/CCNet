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
        whiten_type=['in_nostd'],
        weight_init_scale=0.2,
        with_gc=True,
        with_nl=True,
        nowd=['nl'],
        use_out=False,
        out_bn=False,
    )
)
train_cfg = dict(
    batch_size=16,
    learning_rate=0.01,
    momentum=0.9,
    num_steps=30000,
    power=0.9,
    random_seed=1234,
    restore_from='./dataset/resnet101-imagenet.pth',
    save_num_images=2,
    start_iters=0,
    save_from=29500,
    save_pred_every=100,
    snapshot_dir='snapshots/',
    weight_decay=0.0001
)
data_cfg = dict(
    data_dir='context',
    data_list='./dataset/list/context/train.lst',
    ignore_label=0,
    input_size='520,520',
    num_classes=60,
)
