import os
import numpy as np
from datetime import datetime
import torch.utils.data as data_utils
from sklearn.model_selection import KFold
from attention import Attention, GatedAttention
from BreastLoader import BreastCancerBagsCross
from ColonLoader import ColonCancerBagsCross
import torch.optim as optim
from Procedure import Procedure
from training_report import generate_report, log_results
import argparse
import pandas as pd


def train_model(k=1):
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
        train_loss, train_instance_error, train_bag_error = proc.train(i)
        print('Testing')
        test_loss, test_instance_error, test_bag_error = proc.test()

        test_losses.append(test_loss)
        test_instance_errors.append(test_instance_error)
        test_bag_errors.append(test_bag_error)

        if args.report:
            train_df = pd.DataFrame([[i, train_loss, train_instance_error, train_bag_error]])
            test_df = pd.DataFrame([[i, test_loss, test_instance_error, test_bag_error]])
            log_results(f'{args.output}/train_logs_{k}_{curr_date}', train_df)
            log_results(f'{args.output}/test_logs_{k}_{curr_date}', test_df)

    if args.report:
        pass
        # todo fix generate_report/charts func
        # generate_report(f'{args.output}/train_logs_{k}_{curr_date}')
        # generate_report(f'{args.output}/test_logs_{k}_{curr_date}')


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')

    parser.add_argument('--epochs', type=int, default=1, metavar='N',
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
    parser.add_argument('--nkfcv', type=bool, default=True,
                        help='Disables k-fold cross validation')
    parser.add_argument('--kf', type=int, default=10,
                        help='Number of folds in cross validation')
    parser.add_argument('--report', type=bool, default=True,
                        help='Creates training report')
    parser.add_argument('--output', type=str, default='./TrainingLogs',
                        help='Output directory for logs and charts')

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    test_instance_errors = []
    test_bag_errors = []
    test_losses = []

    if args.ds == 'breast':
        ds_len = 58
    elif args.ds == 'colon':
        ds_len = 100
    else:
        raise NameError(f"dataset {args.ds} not supported")

    idxs = list(range(ds_len))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idxs)

    curr_date = str(datetime.today()).replace(' ', '_')

    if args.nkfcv:
        split_idx = int(args.ttss * ds_len)

        train_idxs = idxs[:split_idx]
        val_idxs = idxs[split_idx:]

        print(f"{split_idx} examples in training set")

        train_model()

        print("Test Bag Error: {}".format(np.mean(test_bag_errors)))
        print("Test Instance Error: {}".format(np.mean(test_instance_errors)))
        print("Test Loss: {}".format(np.mean(test_losses)))
    else:
        cv = KFold(n_splits=args.kf, random_state=args.seed, shuffle=True)

        for i, train_idxs, val_idxs in enumerate(cv.split(idxs)):
            train_model(i + 1)

        # todo std
        print("K-Fold Test Bag Error: {}".format(np.mean(test_bag_errors)))
        print("K-Fold Test Instance Error: {}".format(np.mean(test_instance_errors)))
        print("K-Fold Test Loss: {}".format(np.mean(test_losses)))
