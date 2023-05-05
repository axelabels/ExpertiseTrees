[![DOI](https://zenodo.org/badge/636623487.svg)](https://zenodo.org/badge/latestdoi/636623487)

This repository contains the code to reproduce the results of our paper "Expertise Trees Resolve Knowledge Limitations in Collective Decision-Making". If you use this code in your own research, please cite this paper:

```
@inproceedings{abels2023expertise,
  title={Expertise Trees Resolve Knowledge Limitations in Collective Decision-Making},
      author={Axel Abels and Tom Lenaerts and Vito Trianni and Ann Nowé},
  booktitle={Proceedings of the 40th International Conference on Machine Learning},
  year={2023},
  note={Forthcoming, to be presented in July},
  organization={ICML},
}
```
---------------------------------------
#### Abstract 

Experts advising decision-makers are likely to display expertise which varies as a function of the problem instance. In practice, this may lead to sub-optimal or discriminatory decisions against minority cases. In this work we model such changes in depth and breadth of knowledge as a partitioning of the problem space into regions of differing expertise. We provide here new algorithms that explicitly consider and adapt to the relationship between problem instances and experts' knowledge. We first propose and highlight the drawbacks of a naive approach based on nearest neighbor queries. To address these drawbacks we then introduce a novel algorithm — expertise trees — that constructs decision trees enabling the learner to select appropriate models. We provide theoretical insights and empirically validate the improved performance of our novel approach on a range of problems for which existing methods proved to be inadequate.

# Requirements
Required packages are listed in requirement.txt 

This implementation additionally relies on a slightly modified version of scikit-multiflow provided in the multiflow folder.

# Datasets
Used datasets are either provided in the datasets folder or downloaded from openml.

# Collecting results
Simulations for a given seed can be run as below

```python runner.py seed```

Results will be saved to hdf5 files in the results folder.

# Generating plots
Plots can be generated with the help of the "plots.ipynb" jupyter notebook.
