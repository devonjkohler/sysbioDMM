# sysbioDMM

This repository includes code to model a biological system using a Deep Markov Model. The repository relies on the PPL pyro.

The idea behind this project is to try to infer the interactions between different agents in a biological system using a DMM. These systems generally include features which are hard to model with static enivornments such as DAGs. An example of one of these features is a cycle. Additionally, in a real world experiment we would have limited observations over a certain time period. Therefor rather than employ tatics to deal with a cycle in a DAG, such as unrolling it into more nodes, we can let the DMM learn the relationship.

The code is currently tested on the following biological systems:

1. Lac Operon
2. (More to come)

## References

Jeremy Zucker, Kaushal Paneri, Sara Mohammad-Taheri. Leveraging Structured Biological Knowledge for Counterfactual Inference: a Case Study of Viral Pathogenesis. arXiv:2101.05136

https://pyro.ai/examples/dmm.html

