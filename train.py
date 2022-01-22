import os
import argparse
import logging
import sys

import torch

import datasets
import models
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

from tqdm import tqdm

logger = logging.getLogger()

def parse_arg():
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(asctime)s]: %(levelname)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-dataset', choices=['breast_cancer', 'colon_cancer', 'tiger'], default='breast_cancer')
    parser.add_argument('-batch_size', type=int, default=1)

    parser.add_argument('-feat_dropout', type=float, default=0.3)

    parser.add_argument('-n_tree', type=int, default=50)
    parser.add_argument('-tree_depth', type=int, default=3)
    parser.add_argument('-n_class', type=int, default=2)
    parser.add_argument('-tree_feature_rate', type=float, default=0.5)

    parser.add_argument('-lr', type=float, default=0.001, help="sgd: 10, adam: 0.001")
    parser.add_argument('-gpuid', type=int, default=-1)
    parser.add_argument('-jointly_training', action='store_true', default=True)
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-report_every', type=int, default=10)

    opt = parser.parse_args()
    return opt


def prepare_db(opt):
    print(f"Use %s dataset {opt.dataset}")

    plain_dataset, augmented_dataset = datasets.get_datasets(opt.dataset)

    train_idx, test_idx, _, _ = train_test_split(np.arange(len(augmented_dataset.bags)),
                                                 augmented_dataset.labels,
                                                 test_size=0.1,
                                                 shuffle=True,
                                                 random_state=420)
    return {'dataset': augmented_dataset, 'train_idx': train_idx, 'test_idx': test_idx}


def prepare_models(opt, device, fl_ith_split=3):
    # prepare feature layer
    ds_name, gated, ith_split = opt.dataset, False, fl_ith_split
    path = os.path.join("models", "saved_models", f"{ds_name}_{ith_split}.pt")
    backbone = models.get_model("abmil", ds_name=ds_name, gated=gated)
    backbone.load_state_dict(torch.load(path))
    backbone.eval()
    feature_layer = backbone.feature_extractor.to(device)

    # prepare miforest
    ndf = models.get_model('ndf', n_tree=opt.n_tree, tree_depth=opt.tree_depth, n_in_feature=256,
                           tree_feature_rate=opt.tree_feature_rate, n_class=opt.n_class, jointly_training=opt.jointly_training)
    ndf = ndf.to(device)
    optim = prepare_optim(ndf, opt)
    miforest = models.get_model('mif', device=device, forest=ndf, stop_temp=0.005, optim=optim)

    return feature_layer, miforest


def prepare_optim(model, opt):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=opt.lr, weight_decay=1e-5)


def train(abmil, miforest, device, db, opt):
    # 1. abmil pretrained
    embedded_bags = []
    labels = []

    train_loader = DataLoader(Subset(db['dataset'], db['train_idx']), batch_size=1, shuffle=True)

    for x, y, _ in train_loader:
        x = x.to(device)
        y = y.to(device)
        embedded_bags.append(abmil(x[0]))
        labels.append(y)

    miforest.train(embedded_bags, labels)

    # test
    test_loader = DataLoader(Subset(db['dataset'], db['train_idx']), batch_size=1, shuffle=True)

    embedded_bags_test = []
    labels_test = []

    for x, y, _ in test_loader:
        x = x.to(device)
        y = y.to(device)
        embedded_bags_test.append(abmil(x[0]))
        labels_test.append(y)

    miforest.test(embedded_bags_test, labels_test)


def main():
    log_format = '%(asctime)s [%(levelname)8s] (%(filename)s:%(lineno)s) %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    opt = parse_arg()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Device: {device}')

    db = prepare_db(opt)
    abmil, miforest = prepare_models(opt, device)
    train(abmil, miforest, device, db, opt)


if __name__ == '__main__':
    main()
