import os
import argparse
import collections
import numpy as np
import torch
import torch.nn as nn

import catalyst
from catalyst.utils.dataset import create_dataset, create_dataframe, prepare_dataset_labeling, split_dataframe
from catalyst.utils.pandas import map_dataframe
from catalyst.dl import utils as dutil
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, F1ScoreCallback
from catalyst.data.reader import ImageReader, ScalarReader, ReaderCompose

from senet import se_resnext50_32x4d
from senet_frn import se_resnext50_32x4d_frn
from augmentation import get_transform


"""
Check classification accuracy by FRN
"""


def make_df(data_root):
    dataset = create_dataset(dirs=f"{data_root}/*", extension="*.jpg")
    df = create_dataframe(dataset, columns=["class", "filepath"])

    tag_to_label = prepare_dataset_labeling(df, "class")
    class_names = [name for name, id_ in sorted(tag_to_label.items(), key=lambda x: x[1])]

    df_with_labels = map_dataframe(
        df,
        tag_column="class",
        class_column="label",
        tag2class=tag_to_label,
        verbose=False
    )
    return df_with_labels, class_names


def get_open_fn(data_root, num_class):
    open_fn = ReaderCompose([
        ImageReader(
            input_key="filepath",
            output_key="features",
            datapath=data_root
        ),
        ScalarReader(
            input_key="label",
            output_key="targets",
            default_value=-1,
            dtype=np.int64
        ),
        ScalarReader(
            input_key="label",
            output_key="targets_one_hot",
            default_value=-1,
            dtype=np.int64,
            one_hot_classes=num_class
        )
    ])
    return open_fn


def get_loader(phase, dataset, open_fn, batch_size, num_workers, img_size):
    assert phase in {'train', 'valid'}, f'invalid phase: {phase}'

    transforms_fn = get_transform(phase=phase, img_size=img_size)
    data_loader = dutil.get_loader(
        dataset,
        open_fn=open_fn,
        dict_transform=transforms_fn,
        shuffle=(phase == 'train'),
        drop_last=(phase == 'train'),
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=None
    )
    return data_loader


def get_train_valid_loaders(df, test_size, random_state, data_root,
                            num_class, batch_size, num_workers, img_size):
    open_fn = get_open_fn(data_root=data_root, num_class=num_class)
    args_loader = {
        'open_fn': open_fn,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'img_size': img_size
    }

    train_data, valid_data = split_dataframe(df, test_size=test_size, random_state=random_state)
    train_data, valid_data = train_data.to_dict('records'), valid_data.to_dict('records')

    loaders = collections.OrderedDict()
    loaders["train"] = get_loader(phase='train', dataset=train_data, **args_loader)
    loaders["valid"] = get_loader(phase='valid', dataset=valid_data, **args_loader)
    return loaders


def get_callbacks(num_classes):
    callbacks = [
        AccuracyCallback(num_classes=num_classes),
        F1ScoreCallback(
            input_key="targets_one_hot",
            activation="Softmax"
        )
    ]
    return callbacks


def get_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--gpu', type=str, default='0',
        help='Enable to use specific gpu. You can select multi-gpu(ex. 0,1, if "", CPU is used')
    arg('--frn', action='store_true', help='use FRN if True')
    arg('--seed', type=int, default=42)

    arg('--data-rootdir', type=str, default='./input/artworks/images/images', help='log dir')
    arg('--num-epochs', type=int, default=20)
    arg('--batch-size', type=int, default=16)
    arg('--num-workers', type=int, default=8)
    arg('--img-size', type=int, default=384)
    arg('--fp16', action='store_true', help='use FP16 if True')

    return parser.parse_args()


def main():
    args = get_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    catalyst.utils.set_global_seed(args.seed)
    catalyst.utils.prepare_cudnn(deterministic=True)

    print('Make Data set data frame')
    df, class_names = make_df(data_root=args.data_rootdir)
    num_class = len(class_names)

    print('Get data loaders')
    loaders = get_train_valid_loaders(
        df=df,
        test_size=0.2,
        random_state=args.seed,
        data_root=args.data_rootdir,
        num_class=num_class,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )

    print('Make model')
    if args.frn:
        model = se_resnext50_32x4d_frn()
    else:
        model = se_resnext50_32x4d()
    model.last_linear = nn.Linear(512 * 16, num_class)

    print('Get optimizer and scheduler')
    # learning rate for FRN is very very sensitive !!!
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4 if args.frn else 3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.num_epochs,
        eta_min=3e-6 if args.frn else 1e-5,
        last_epoch=-1
    )

    log_base = './output/cls'
    dir_name = f'seresnext50{"_frn" if args.frn else ""}_bs_{args.batch_size}_fp16_{args.fp16}'

    print('Start training...')
    runner = SupervisedRunner(device=catalyst.utils.get_device())
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=os.path.join(log_base, dir_name),
        callbacks=get_callbacks(num_classes=num_class),
        num_epochs=args.num_epochs,
        main_metric="accuracy01",
        minimize_metric=False,
        fp16=dict(opt_level="O1") if args.fp16 else None,
        verbose=False
    )


if __name__ == '__main__':
    main()
