Welcome to the documentation of "NPDoseResponse"!
===================================

**NPDoseResponse** is a Python library for estimating and conducting valid inference on a (causal) dose-response curve and its derivative function via novel integral and localized derivative estimators.

A Preview into the Proposed Methodology
------------

The proposed debiasing method introduces a novel debiased estimator for inferring the linear regression function with "missing at random (MAR)" outcomes. The key idea is to correct the bias of the Lasso solution [2]_ with complete-case data through a quadratic debiasing program with box constraints and construct the confidence interval based on the asymptotic normality of the debiased estimator.

More details can be found in :doc:`Methodology <method>` and the reference paper [1]_.

.. note::

   This project is under active development.
   

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   api_reference
   
   
References
----------
.. [1] Yikun Zhang, Yen-Chi Chen, Alexander Giessing (2024+) Nonparametric Inference on Dose-Response Curves Without the Positivity Condition.


