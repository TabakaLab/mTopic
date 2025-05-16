mTopic - Multimodal topic modeling for single-cell data
=======================================================

`mTopic` is an open-source Python library with computational tools for modeling multimodal topics in single-cell data.

.. raw:: html

    <div class="mTopic-logo-container">
      <img class="logo-light"
           src="_static/mTopic_logo_light_background.png"
           alt="mTopic logo (light)">
      <img class="logo-dark"
           src="_static/mTopic_logo_dark_background.png"
           alt="mTopic logo (dark)">
    </div>

    <script>
      function swapLogo() {
        const theme = document.documentElement.getAttribute('data-theme');
        const isDark = theme === 'dark';
        const lightImg = document.querySelector('.mTopic-logo-container .logo-light');
        const darkImg  = document.querySelector('.mTopic-logo-container .logo-dark');
        if (lightImg && darkImg) {
          lightImg.style.display = isDark ? 'none'  : 'block';
          darkImg .style.display = isDark ? 'block' : 'none';
        }
      }

      // Initial swap after all scripts run
      window.addEventListener('load', swapLogo);
      // Watch for theme‐toggle changes
      new MutationObserver(swapLogo)
        .observe(document.documentElement, {
          attributes: true,
          attributeFilter: ['data-theme', 'data-bs-theme']
        });
    </script>

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

.. toctree::
   :caption: Documentation
   :hidden:

   installation
   notebooks/T1_data_preprocessing
   notebooks/T2_tonsils
   notebooks/T3_pbmc
   api
