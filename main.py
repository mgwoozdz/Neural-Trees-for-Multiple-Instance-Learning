import glob
import numpy as np
import torch
import torch.utils.data as data_utils
from attention import Attention
from BreastLoader import BreastCancerBagsCross
from ColonLoader import ColonCancerBagsCross
import torch.optim as optim
from Procedure import Procedure


#Breast Cancer Training
'''
if __name__ == "__main__":
    train_loader = data_utils.DataLoader(BreastCancerBagsCross("Data/BreastCancer/", [i for i in range(40)],[i for i in range(40,58)], train=True))
    test_loader = data_utils.DataLoader(BreastCancerBagsCross("Data/BreastCancer/", [i for i in range(40)],[i for i in range(40,58)], train=False))
    lr = 10e-4
    reg = 10e-5
    model = Attention()
    optimizer = optim.Adam(model.parameters(), lr = lr, betas=(0.9, 0.999), weight_decay= reg)
    #procedur is in BCProcedur.py file
    proc = Procedure(model, train_loader, test_loader, optimizer = optimizer)

    print('Start Training')
    for i in range(30):
        proc.train(i)

    proc.test()
'''

#Colon Cancer Training
if __name__ == "__main__":
    train_loader = data_utils.DataLoader(
        ColonCancerBagsCross("Data/ColonCancer/", [i for i in range(40)], [i for i in range(40, 100)], train=True))
    test_loader = data_utils.DataLoader(
        ColonCancerBagsCross("Data/ColonCancer/", [i for i in range(40)], [i for i in range(40, 100)], train=False))

    lr = 10e-4
    reg = 10e-5
    model = Attention()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=reg)
    proc = Procedure(model, train_loader, test_loader, optimizer=optimizer)

    print('Start Training')
    for i in range(30):
       proc.train(i)

    proc.test()
