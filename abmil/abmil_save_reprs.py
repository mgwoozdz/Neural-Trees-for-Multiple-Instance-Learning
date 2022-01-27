from models.abmil import SAVE_PATH, AttentionBasedMIL
from utils.colon_cancer import ColonCancerDataset
from utils.helpers import get_device
import torch as t

device = get_device()
print(f"Device: {device}")

dataset = ColonCancerDataset(root_dir="datasets/colon_cancer")

model = AttentionBasedMIL(return_repr=True)
model = model.to(device)
t.load(SAVE_PATH, map_location=device)
model.eval()

dataset.save_reprs_to_folder(model, folder="datasets/colon_cancer_reprs")
