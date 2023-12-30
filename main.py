import os
import torch
import utils
import time
import wandb
import numpy as np
import torch.utils.data as data

from models import WSAD
from datasets import get_dataset
from options import parse_args
from test import test, get_test_bz
from losses import LossComputer
from train import train
from tqdm import tqdm
from collections import defaultdict

localtime = time.localtime()
time_ymd = time.strftime("%Y-%m-%d", localtime)

def init_wandb(args):
    if 'xd' in  args.dataset:
        datasetname = 'xd-violence'
    elif 'ucf' in args.dataset:
        datasetname = 'ucf-crime'
    else:
        raise RuntimeError(f'Unknown Dataset: {args.dataset}')
    
    wandb.init(
        project=f"ECD-{args.model_type}-{args.dataset}",
        name=args.version,
        config={
            'optimization:lr':args.lr,
            'optimization:iters': args.max_iter,

            'dataset:dataset': datasetname,
            'decouple:w_bag': args.w_bag,
        },
        settings=wandb.Settings(code_dir=os.path.dirname(os.path.abspath(__file__))),
        save_code=True,
    )


if __name__ == "__main__":
    args = parse_args()
    
    args.log_path = os.path.join(args.log_path, time_ymd, args.dataset, args.version)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
    init_wandb(args)

    worker_init_fn = None

    if args.seed >= 0:
        utils.set_seed(args.seed)
        worker_init_fn = np.random.seed(args.seed)

    net = WSAD(args.in_channel, flag= "Train", args=args)
    utils.save_settings(args.log_path, args, net)
    net = net.cuda()

    dataset = get_dataset(args.dataset)

    normal_train_loader = data.DataLoader(
        dataset(root_dir=args.root_dir, mode='Train', num_segments=args.num_segments, len_feature=args.in_channel, is_normal=True),
        batch_size=args.batch_nor,
        shuffle=True, num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    abnormal_train_loader = data.DataLoader(
        dataset(root_dir=args.root_dir, mode='Train', num_segments=args.num_segments, len_feature=args.in_channel, is_normal=False),
        batch_size=args.batch_abn,
        shuffle=True, num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    test_loader = data.DataLoader(
        dataset(root_dir=args.root_dir, mode='Test', num_segments=args.num_segments, len_feature=args.in_channel, is_normal=False),
        batch_size=get_test_bz(args.dataset),
        shuffle=False, num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
    )

    criterion = LossComputer(args)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas = (0.9, 0.999), weight_decay = args.weight_decay)

    best_scores = defaultdict(lambda: -1)

    metrics, out_scores = test(net, test_loader, args.dataset)

    for score_name, score in out_scores.items():
        best_name = f'best_{score_name}'
        best_scores[best_name] = score if score > best_scores[best_name] else best_scores[best_name]

    for step in tqdm(
        range(1, args.max_iter + 1),
        total = args.max_iter,
        dynamic_ncols = True
    ):
        if (step - 1) % len(normal_train_loader) == 0:
            normal_loader_iter = iter(normal_train_loader)

        if (step - 1) % len(abnormal_train_loader) == 0:
            abnormal_loader_iter = iter(abnormal_train_loader)

        losses = train(net, normal_loader_iter, abnormal_loader_iter, optimizer, criterion, step)

        wandb.log(losses, step=step)
        if step % args.plot_freq == 0 and step > 0:
            metrics, out_scores = test(net, test_loader, args.dataset)
        
            utils.save_best(metrics, best_scores, net, args.dataset, step, args.metric, args.log_path, args.model_path)

            for score_name, score in out_scores.items():
                best_name = f'best_{score_name}'
                best_scores[best_name] = score if score > best_scores[best_name] else best_scores[best_name]

        wandb.log(out_scores, step=step)
        wandb.log(best_scores, step=step)