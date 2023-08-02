# Meta-Learning for Scalable Hydrological Streamline Delineation

## Authors

- Nattapon Jaroenchai <sup>a, b</sup>
- Shaowen Wang <sup>a, b, *</sup>
- Zhaonan Wang <sup>a, b</sup>
- Lawrence V. Stanislawski <sup>c</sup>
- Ethan Shavers <sup>c</sup>

<sup>a</sup> Department of Geography and Geographic Information Science, University of Illinois at Urbana-Champaign, Urbana, IL, USA
<sup>b</sup> CyberGIS Center for Advanced Digital and Spatial Studies, University of Illinois at Urbana-Champaign, Urbana, IL, USA
<sup>c</sup> U.S. Geology Survey, Center of Excellence for Geospatial Information Science, Rolla, MO, USA
<sup>d</sup> School of Geoscience and Info-Physics, Central South University, Changsha, Hunan, China

## Introduction

This repository contains the code for the research project titled "Meta-Learning for Scalable Hydrological Streamline Delineation." The project aims to enhance the accuracy of streamline network delineation for hydrological applications using advanced machine learning techniques. Specifically, it explores the application of Meta-Learning (MAML) and Meta-Learning with Spatial-Temporal Memory (MAML-w-mem) to improve model performance in dealing with different geographical regions.

The code in this repository is a work in progress, and it contains two main folders:

1. **MAML**: This folder contains the implementation of the MAML algorithm for meta-learning in the context of streamline network delineation. MAML leverages knowledge from multiple source tasks to improve the performance of the target task, offering promise in enhancing machine learning models' transferability.

2. **MAML-w-mem**: This folder contains the implementation of the MAML algorithm with the addition of Spatial-Temporal Memory (ST-Mem) for streamline network delineation. The ST-Mem module is designed to capture long-term temporal information and patterns, further enhancing the prediction performance of the model.

Please note that as the code is still a work in progress, certain functionalities and optimizations may be ongoing. Additionally, the provided datasets and models might be incomplete or placeholders for future updates.

## Folder Structure

The repository's folder structure is as follows:

- `MAML/`: This directory contains the MAML implementation for streamline network delineation.

- `MAML-w-mem/`: This directory contains the MAML implementation with the Spatial-Temporal Memory (ST-Mem) module for streamline network delineation.

- `data/`: This directory (not present in the current version) will contain the necessary datasets used for the streamline delineation task. Due to data size limitations, the complete dataset may not be provided, but instructions on how to access or preprocess the data will be included.

- `results/`: This directory (not present in the current version) will store the model outputs, evaluation metrics, and any visualization results generated during the experiments.

- `docs/`: This directory (not present in the current version) will contain any additional documentation, papers, or presentations related to the project.

## Getting Started

As the code is a work in progress, detailed instructions for getting started with each folder (MAML and MAML-w-mem) will be provided within the respective folders. This will include information on data preprocessing, model configurations, and how to execute the code for experiments.

## Citation

As this work is ongoing and currently under development, citation details for the research project will be provided in future updates when the code and research are ready for publication.

## Contact Information

For any inquiries or questions about the project, please contact the corresponding author:

- Shaowen Wang: *(Include email address or contact details here)*

We appreciate your interest in our research, and we look forward to sharing the completed code and results with the community soon!