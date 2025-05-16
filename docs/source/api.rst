API
===

Import `mTopic`:

.. code-block:: python

    import mtopic

Core submodules
---------------

Data reading (`read`)
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated

   mtopic.read.h5mu

Preprocessing (`pp`)
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated

   mtopic.pp.permute
   mtopic.pp.tfidf
   mtopic.pp.clr
   mtopic.pp.scale_counts
   mtopic.pp.filter_var_knee
   mtopic.pp.filter_var_list
   mtopic.pp.feature_associations_data

Tools (`tl`)
^^^^^^^^^^^^

.. autosummary::
   :toctree: generated

   mtopic.tl.MTM
   mtopic.tl.sMTM
   mtopic.tl.export_params
   mtopic.tl.zscores
   mtopic.tl.umap
   mtopic.tl.feature_associations

Plotting (`pl`)
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated

   mtopic.pl.filter_var_knee
   mtopic.pl.filter_topics
   mtopic.pl.topics
   mtopic.pl.scatter_pie
   mtopic.pl.dominant_topics
   mtopic.pl.signatures
   mtopic.pl.zscores
   mtopic.pl.corr_heatmap
   mtopic.pl.feature_activity
