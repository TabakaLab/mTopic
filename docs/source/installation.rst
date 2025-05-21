Installation Guide
==================

We recommend installing ``mTopic`` in a fresh environment to avoid dependency conflicts.

Prerequisites
-------------

- **Conda**: `Miniconda or Anaconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_
- **venv**: Python 3.11+

Choose one of the following options:

Option A: Conda (recommended)
-----------------------------

.. code-block:: bash

   git clone https://github.com/TabakaLab/mTopic.git
   cd mTopic
   conda create -n mtopic python=3.11.2
   conda activate mtopic
   pip install .
   conda deactivate  # optional

Option B: venv
--------------

.. code-block:: bash

   git clone https://github.com/TabakaLab/mTopic.git
   cd mTopic
   python3 -m venv mtopic-env
   source mtopic-env/bin/activate
   pip install .
   deactivate  # optional
