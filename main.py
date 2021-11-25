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
from training_report import log_results
import argparse
import pandas as pd
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


def train_model(idx=1):
    if args.ds == 'breast':
        train_loader = data_utils.DataLoader(BreastCancerBagsCross("Data/BreastCancer/",
                                                                   train_idxs,
                                                                   val_idxs,
                                                                   train=True))
        val_loader = data_utils.DataLoader(BreastCancerBagsCross("Data/BreastCancer/",
                                                                 train_idxs,
                                                                 val_idxs,
                                                                 train=False))
    elif args.ds == 'colon':
        train_loader = data_utils.DataLoader(ColonCancerBagsCross("Data/ColonCancer/",
                                                                  train_idxs,
                                                                  val_idxs,
                                                                  train=True))
        val_loader = data_utils.DataLoader(ColonCancerBagsCross("Data/ColonCancer/",
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
    proc = Procedure(model, train_loader, val_loader, optimizer=optimizer)

    log_train_filename = f"{args.output}/train_logs_{idx}_{curr_date}"
    log_val_filename = f"{args.output}/val_logs_{idx}_{curr_date}"
    for epoch in tqdm(
            range(args.epochs),
            desc=f"training model on split {idx} of {args.folds}" if args.folds else "training model",
            unit="epoch"
    ):
        train_loss, train_error = proc.train()
        print(f"Epoch: {epoch}, Loss: {train_loss:.4f}, Error: {train_error:.4f}")
        val_loss, val_error = proc.test()
        print(f"Epoch: {epoch}, Val. Loss: {val_loss:.4f}, Val. Error: {val_error:.4f}")
        if args.report:
            train_df = pd.DataFrame([[epoch, train_loss, train_error]])
            val_df = pd.DataFrame([[epoch, val_loss, val_error]])
            log_results(log_train_filename, train_df)
            log_results(log_val_filename, val_df)

    if args.report:
        pass
        # todo fix generate_report/charts func
        # figure_filename_prefix = f"model_{idx}"
        # generate_report(log_train_filename, log_test_filename, figure_filename_prefix)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST bags Example")

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
    parser.add_argument('--no_kfold_cv', dest='kfold_cv', action='store_false', help='Disables k-fold cross validation')
    parser.set_defaults(kfold_cv=True)
    parser.add_argument('--folds', type=int, default=10,
                        help='Number of folds in cross validation')
    parser.add_argument('--report', type=bool, default=True,
                        help='Creates training report')
    parser.add_argument('--output', type=str, default='./TrainingLogs',
                        help='Output directory for logs and charts')

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.ds == "breast":
        ds_len = 58
    elif args.ds == "colon":
        ds_len = 100
    else:
        raise NameError(f"dataset {args.ds} not supported")

    idxs = list(range(ds_len))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idxs)

    curr_date = str(datetime.today()).replace(" ", "_")

    if args.kfold_cv:
        print("(debug) using k-fold cross validation")
        cv = KFold(n_splits=args.folds, random_state=args.seed, shuffle=True)

        for i, splitted_idxs in enumerate(cv.split(idxs)):
            train_idxs = splitted_idxs[0]
            val_idxs = splitted_idxs[1]
            train_model(idx=i+1)

    else:
        print("(debug) not using k-fold cross validation")
        split_idx = int(args.ttss * ds_len)

        train_idxs = idxs[:split_idx]
        val_idxs = idxs[split_idx:]

        print(f"{split_idx} examples in training set")

        train_model()
