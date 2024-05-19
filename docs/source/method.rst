Methodology
===========

Model Setup
------------

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

Given the above identification formula, one traditional method for estimating the dose-response curve :math:`m(t)` is through the following regression adjustment (RA) or G-computation estimator

.. math::

    \hat{m}_{RA}(t)  = \frac{1}{n}\sum_{i=1}^n \hat{\mu}(t,\textbf{S}_i),

where :math:`\hat{\mu}(t,\textbf{s})` is any consistent estimator of the conditional mean outcome function :math:`\mu(t,\textbf{s})`. However, when the positivity condition does not hold, the regression adjustment estimator will be unstable and even inconsistent. This is because without the positivity condition, the joint density :math:`p(t,\textbf{S}_i)=p(t|\textbf{S}_i)\cdot p_S(\textbf{S}_i)` could be closer to 0 for some :math:`i=1,...,n`.


Key Insights and Proposed Estimators
------------------------------------

To bypass the strong positivity condition, we consider imposing the following key assumption:

.. math::

    \mathbb{E}\left[\mu(T,\textbf{S})\right]=\mathbb{E}\left[m(T)\right] \quad \text{ and } \quad \theta(t)=\mathbb{E}\left[\frac{\partial}{\partial t} \mu(t,\textbf{S})\right] 
= \mathbb{E}\left[\frac{\partial}{\partial t} \mu(t,\textbf{S}) \Big|T=t\right].

It can be verified that the additive confounding model with :math:`\mu(T,\textbf{S})=m(T)+\eta(\textbf{S})` satisfies the above two equalities.

Under the above assumption, we construct our estimator of :math:`m(t)` from three critical insights:

* **Insight 1:** :math:`\mu(t,\textbf{s})` and :math:`\frac{\partial}{\partial t}\mu(t,\textbf{s})` can be consistently estimated at each observation :math:`(T_i,\textbf{S}_i)` for :math:`i=1,...,n`.

* **Insight 2:** :math:`\theta(t)` can be consistently estimated through the localized form :math:`\theta_C(t)=\mathbb{E}\left[\frac{\partial}{\partial t} \mu(t,\textbf{S}) \big|T=t\right]`.

* **Insight 3:** By the fundamental theorem of calculus, we know that

.. math::

    m(t) = m(T) + \int_{\tilde{t}=T}^{\tilde{t}=t} m'(\tilde{t})\, d\tilde{t} = m(T)+ \int_{\tilde{t}=T}^{\tilde{t}=t} \theta(\tilde{t})\, d\tilde{t}.

Under our key assumption, we can take the expectation on both sides of the above equality to obtain that

.. math::

    m(t) = \mathbb{E}\left[m(T) + \int_{\tilde{t}=T}^{\tilde{t}=t} \theta(\tilde{t})\, d\tilde{t}\right] =\mathbb{E}\left[\mu(T,\textbf{S})\right] + \mathbb{E}\left[\int_{\tilde{t}=T}^{\tilde{t}=t} \theta_C(\tilde{t})\, d\tilde{t}\right] = \mathbb{E}(Y) + \mathbb{E}\left\{\int_{\tilde{t}=T}^{\tilde{t}=t} \mathbb{E}\left[\frac{\partial}{\partial t}\mu(\tilde{t},\textbf{S})\Big|T=\tilde{t}\right] \, d\tilde{t}\right\}

Based on the above three insights, we thus propose an *integral estimator* of the dose-response curve :math:`m(t)` as:

.. math::

    \hat{m}_\theta(t) = \frac{1}{n}\sum_{i=1}^n \left[Y_i + \int_{\tilde{t}=T_i}^{\tilde{t}=t} \hat{\theta}_C(\tilde{t})\, d\tilde{t} \right],

where :math:`\hat{\theta}_C(t)` is a consistent estimator of :math:`\theta_C(t) = \mathbb{E}\left[\frac{\partial}{\partial t}\mu(t,\textbf{S})\big|T=t\right] = \int \frac{\partial}{\partial t} \mu(t,\textbf{s})\, d\mathrm{P}(\textbf{s}|t)`. The estimator :math:`\hat{\theta}_C(t)` of the derivative effect :math:`\theta(t)` includes two nuisance functions:

* We fit the partial derivative :math:`\beta_2(t,\textbf{s})=\frac{\partial}{\partial t} \mu(t,\textbf{s})` of the conditional mean outcome function by (partial) local polynomial regression;

* We estimate the conditional cumulative distribution function (CDF) :math:`\mathrm{P}(\textbf{s}|t)` via Nadaraya-Watson conditional CDF estimator.

This leads to our proposed localized derivative estimator of :math:`\theta(t)` as:

.. math::

    \hat{\theta}_C(t)= \frac{\sum_{i=1}^n \hat{\beta}_2(t,\textbf{S}_i) \cdot \bar{K}_T\left(\frac{T_i-t}{\hslash}\right)}{\sum_{j=1}^n \bar{K}_T\left(\frac{T_j-t}{\hslash}\right)},

where :math:`\bar{K}_T:\mathbb{R}\to[0,\infty)` is a kernel function.


Fast Computing Algorithm
----------------------------


Bootstrap Inference
----------------------------


References
----------

.. [1] Yikun Zhang, Alexander Giessing, Yen-Chi Chen (2024+). Nonparametric Inference on Dose-Response Curves Without the Positivity Condition.
