mTopic – Multimodal Topic Modeling for Single-Cell Data
=======================================================

.. image:: docs/source/_static/mTopic_logo_light_background.png
   :align: center
   :width: 800px
   :alt: mTopic logo

mTopic is a scalable Python package for topic modeling on multimodal single-cell datasets. It supports spatial and non-spatial data, enabling integrated analysis across multiple molecular layers, such as RNA, ATAC, and protein expression.

By capturing shared patterns across modalities, mTopic helps reveal latent biological structures, regulatory programs, and spatial organization within complex tissues.

Resources
---------

- GitHub (Python) (https://github.com/TabakaLab/mTopic)  
- GitHub (R companion) (https://github.com/TabakaLab/mTopicR)  
- Documentation (https://mtopic.readthedocs.io/)

Features
--------

- Supports multimodal data integration (RNA, ATAC, protein, histone modifications)
- Handles spatial and non-spatial single-cell data
- Implements scalable variational inference for efficient training
- Allows per-modality preprocessing, signature detection, and z-score enrichment
- Built on top of the MuData (https://muon.readthedocs.io/) data structure (`.h5mu` format)
- Visualization tools for topic distributions, dominant topic maps, and feature signatures
