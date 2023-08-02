# MAML With ST-Mem Module

## Description

This folder contains the implementation of Model-Agnostic Meta-Learning (MAML) with the Spatial-Temporal Memory (ST-Mem) module. The research paper "Learning from Multiple Cities: A Meta-Learning Approach for Spatial-Temporal Prediction" by Huaxiu Yao, Yiding Liu, Ying Wei, Xianfeng Tang, and Zhenhui Li serves as the basis for this implementation.

## Research Paper Overview

The research paper addresses the problem of spatial-temporal prediction for smart cities, which has applications in traffic control, taxi dispatching, environmental policy making, and other urban planning tasks. The data collected from different cities often have unbalanced spatial distributions, with some cities having data for extended periods and others having data for only a short period. The paper proposes a meta-learning approach that leverages information from multiple cities to improve the stability of knowledge transfer. Instead of transferring knowledge from a single source city to a target city, they utilize data from multiple cities to enhance spatial-temporal prediction.

The proposed model combines a spatial-temporal network with the meta-learning paradigm. The meta-learning paradigm enables the learning of a well-generalized initialization of the spatial-temporal network that can be effectively adapted to different target cities. Additionally, a pattern-based spatial-temporal memory is incorporated to capture long-term temporal information, such as periodicity, which further enhances prediction performance.

## Contents

The contents of this folder are as follows:

- **(To be added later)** `maml_stmem.py`: This script will contain the implementation of MAML with the ST-Mem module based on the research paper.

## Contribution

This repository is actively being developed, and contributions are welcome. If you are interested in adding the code for MAML with the ST-Mem module based on the research paper or making any other enhancements, feel free to fork this repository and submit a pull request.

## Future Updates

We are actively working on implementing MAML with the ST-Mem module based on the research paper "Learning from Multiple Cities: A Meta-Learning Approach for Spatial-Temporal Prediction." Once the implementation is ready, the repository will be updated with the relevant script and further instructions on how to use it.

## References

For more information about the research paper "Learning from Multiple Cities: A Meta-Learning Approach for Spatial-Temporal Prediction" by Huaxiu Yao, Yiding Liu, Ying Wei, Xianfeng Tang, and Zhenhui Li, please refer to the original paper.

As this work is based on the research paper, the implementation aims to follow the methodology and ideas described in the paper to the best extent possible.

Please stay tuned for updates! If you have any questions or suggestions, feel free to reach out. Thank you for your interest in this project.