import glob
import numpy as np
import torch
import torch.utils.data as data_utils
from attention import Attention, GatedAttention
from BreastLoader import BreastCancerBagsCross
from ColonLoader import ColonCancerBagsCross
import torch.optim as optim
from Procedure import Procedure
import argparse


if __name__ == "__main__":

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')

    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=10e-4, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--wd', type=float, default=10e-5, metavar='R',
                        help='weight decay')
    parser.add_argument('--seed', type=int, default=420, metavar='S',
                        help='random seed (default: 420)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--model', type=str, default='attention',
                        help='Choose b/w attention and gated_attention')
    parser.add_argument('--ds', type=str, default='colon',
                        help='Choose b/w colon and breast')
    parser.add_argument('--ttss', type=float, default=0.7,
                        help='Percentage of train/test split size (default: 0.7)')

    args = parser.parse_args()

    if args.ds == 'breast':
        ds_len = 58
    elif args.ds == 'colon':
        ds_len = 100
    else:
        raise NameError(f"dataset {args.ds} not supported")

    idxs = list(range(ds_len))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idxs)

    split_idx = int(args.ttss * ds_len)

    train_idxs = idxs[:split_idx]
    val_idxs = idxs[split_idx:]

    print(f"{split_idx} examples in training set")

    if args.ds == 'breast':
        train_loader = data_utils.DataLoader(BreastCancerBagsCross("Data/BreastCancer/",
                                                                   train_idxs,
                                                                   val_idxs,
                                                                   train=True))
        test_loader = data_utils.DataLoader(BreastCancerBagsCross("Data/BreastCancer/",
                                                                  train_idxs,
                                                                  val_idxs,
                                                                  train=False))
    elif args.ds == 'colon':
        train_loader = data_utils.DataLoader(ColonCancerBagsCross("Data/ColonCancer/",
                                                                  train_idxs,
                                                                  val_idxs,
                                                                  train=True))
        test_loader = data_utils.DataLoader(ColonCancerBagsCross("Data/ColonCancer/",
                                                                 train_idxs,
                                                                 val_idxs,
                                                                 train=False))
    else:
        raise NameError(f"dataset {args.ds} not supported")

    if args.model == "attention":
        model = Attention()
    elif args.mnodel == "gated_attention":
        model = GatedAttention()
    else:
        raise NameError(f"model {args.model} not supported")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
    proc = Procedure(model, train_loader, test_loader, optimizer=optimizer)

    print('Start Training')
    for i in range(args.epochs):
       proc.train(i)
    print('Testing')
    proc.test()
