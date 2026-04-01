Installation Guide
==================

We recommend installing ``mTopic`` in a fresh environment to avoid dependency conflicts.

Prerequisites
-------------

- **Conda**: `Miniconda or Anaconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_
- **Python**: 3.11+

Create environment
------------------

.. code-block:: bash

   conda create -n mtopic python=3.11 -y
   conda activate mtopic
   python -m pip install -U pip

Install PyTorch
---------------

``mTopic`` requires PyTorch, which must be installed before installing ``mTopic``.

PyTorch wheels already include the CUDA runtime, so a local CUDA toolkit is not required.

**NVIDIA GPU (recommended)**

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cu124

This installs the CUDA 12.4 build. If your GPU is not detected or you encounter CUDA/driver errors, install a build compatible with your system instead (e.g., ``cu121`` or ``cu118``).

You can check your NVIDIA driver and CUDA compatibility with:

.. code-block:: bash

   nvidia-smi

Browse all available compute platforms (``cpu``, ``cu118``, ``cu121``, ``cu124``, ``cu126``, etc.) at `https://download.pytorch.org/whl/ <https://download.pytorch.org/whl/>`_.

**CPU-only (no GPU)**

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cpu

Install mTopic
--------------

.. code-block:: bash

   git clone https://github.com/TabakaLab/mTopic.git
   pip install mTopic/.

Verification
------------

After installation, verify that PyTorch works correctly:

.. code-block:: bash

   python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())"

- ``True`` - GPU is active
- ``False`` - running on CPU

Jupyter notebook kernel
-----------------------

To create a Jupyter notebook kernel, run:

.. code-block:: bash

   python -m ipykernel install --user --name mtopic --display-name "Python (mTopic)"
