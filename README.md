# Multiple Instance Neural Decision Forest - MINDF

Assignment project for Computer Vision course, UJ 2021/22

The idea is to implement [MIForest](https://link.springer.com/content/pdf/10.1007%2F978-3-642-15567-3_3.pdf)
replacing random forest with [NDF](https://openaccess.thecvf.com/content_iccv_2015/papers/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.pdf) 
using embeddings obtained by [AbMIL](https://arxiv.org/pdf/1802.04712.pdf).

Focus is put on solving two histopathological ([breast cancer]() and [colon cancer]() datasets) 
and classics ([elephant, fox, musk1, musk2, tiger datasets]()) benchmarks.

## Experiments

We conducted (in some cases at least tried to) following experiments:

- Reproducing AbMIL results on histopathological benchmark (run`python reproduce_abmil.py`).
- Reproducing MIForest results on classics benchmark (`python reproduce_miforest.py`).
- Introducing NDF on histopathological  benchmark (`python mindf.py`).

Check `report.pdf` file for discussion.




