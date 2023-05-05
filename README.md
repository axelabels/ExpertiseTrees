# ExpertiseTrees


This repository contains the code to reproduce the results of our paper "Expertise Trees Resolve Knowledge Limitations in Collective Decision-Making". If you use this code in your own research, please cite this paper:

```
@article{abels2023expertise,
      title={Expertise Trees Resolve Knowledge Limitations in Collective Decision-Making}, 
      author={Axel Abels and Tom Lenaerts and Vito Trianni and Ann Nowé},
      year={2023},
      eprint={2305.01063},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
---------------------------------------
#### Abstract 

Experts advising decision-makers are likely to display expertise which varies as a function of the problem instance. In practice, this may lead to sub-optimal or discriminatory decisions against minority cases. In this work we model such changes in depth and breadth of knowledge as a partitioning of the problem space into regions of differing expertise. We provide here new algorithms that explicitly consider and adapt to the relationship between problem instances and experts' knowledge. We first propose and highlight the drawbacks of a naive approach based on nearest neighbor queries. To address these drawbacks we then introduce a novel algorithm — expertise trees — that constructs decision trees enabling the learner to select appropriate models. We provide theoretical insights and empirically validate the improved performance of our novel approach on a range of problems for which existing methods proved to be inadequate.
