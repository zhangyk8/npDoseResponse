Welcome to the documentation of "NPDoseResponse"!
===================================

**NPDoseResponse** is a Python library for estimating and conducting valid inference on a (causal) dose-response curve and its derivative function via novel integral and localized derivative estimators.

A Preview into the Proposed Methodology
------------

Existing methods in causal inference for continuous treatments often rely on the particularly strong positivity condition. We propose a novel integral estimator of the causal effects with continuous treatments (i.e., dose-response curves) without requiring the positivity condition. Our approach involves estimating the derivative function of the treatment effect on each observed data sample and integrating it to the treatment level of interest so as to address the bias resulting from the lack of positivity condition. Valid inferences on the dose-response curve and its derivative function can also be conducted with our proposed estimators via bootstrap methods.

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

