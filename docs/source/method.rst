Methodology
===========

Consider a random sample :math:`\{(Y_i,T_i,\textbf{S}_i)\}_{i=1}^n \subset \mathbb{R}\times \mathbb{R} \times \mathbb{R}^d` generated from the following model:

.. math::

    Y=\mu(T,\textbf{S})+\epsilon \quad \text{ and } \quad T=f(\textbf{S})+E,

where 

* :math:`Y` is the outcome variable,
* :math:`T` is the continuous treatment variable,
* :math:`\textbf{S}` is a vector of covariates that incorporate confounding variables,
* :math:`E\in\mathbb{R}` is the treatment variation with :math:`\mathbb{E}(E)=0` and :math:`E` being independent of :math:`\textbf{S}`,
* :math:`\epsilon\in\mathbb{R}` is an exogenous noise variable with :math:`\mathbb{E}(\epsilon)=0, \mathrm{Var}(\epsilon)=\sigma^2>0,\mathrm{E}(\epsilon^4)<\infty`.

Let :math:`Y(t)` be the potential outcome that would have been observed under treatment level :math:`T=t`. Under some identification conditions (see Section 2.2 in our paper [1]_), the (causal) dose-response curve :math:`t\mapsto m(t)=\mathbb{E}\left[Y(t)\right]` coincides with the covariate-adjusted regression function :math:`t\mapsto \mathbb{E}\left[\mu(t,\textbf{S})\right]` and can thus be identified from the observed data :math:`\{(Y_i,T_i,\textbf{S}_i)\}_{i=1}^n`. In addition, we also consider estimating the derivative effect :math:`\theta(t)=m'(t)=\frac{d}{dt}\mathbb{E}\left[\mu(t,\textbf{S})\right]`.




References
----------

.. [1] Yikun Zhang, Alexander Giessing, Yen-Chi Chen (2024+). Nonparametric Inference on Dose-Response Curves Without the Positivity Condition.
