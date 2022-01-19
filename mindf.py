"""
We want to train NDF with MIForest methodology on top of ABMIL (pretrained with attention) embeddings.

(keep_abmil) use ABMIL's feature extractor as NDF's feature layer
    we do not need to store large amount of data
    we can make use of data augmentations
    we can further keep it frozen or finetune

(drop_abmil) use ABMIL to transform dataset into its embedded counterpart and feed NDF (with no feature layer) it
    faster since data augmentations are bottleneck
"""
import os
import torch
import datasets
import models


def keep_abmil():

    # load entire abmil
    ds_name, gated, ith_split = "breast_cancer", False, 3
    path = os.path.join("models", "saved_models", f"{ds_name}_{ith_split}.pt")
    backboone = models.get_model("abmil", ds_name=ds_name, gated=gated)
    backboone.load_state_dict(torch.load(path))
    print(backboone)

    # keep only its feature extractor
    # TODO

    # instantiate ndf
    # TODO


if __name__ == "__main__":
    keep_abmil()
