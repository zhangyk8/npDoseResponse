Methodology II (DR Inference With and Without Positivity)
===========

Identification and Estimation Under Positivity
------------

Let :math:`Y(t)` be the potential outcome that would have been observed under treatment level :math:`T=t`. Consider a random sample :math:`\{(Y_i,T_i,\textbf{S}_i)\}_{i=1}^n \subset \mathbb{R}\times \mathbb{R} \times \mathbb{R}^d`. We assume the following basic identification conditions.

* **A1(a) (Consistency)** :math:`T_i=t` implies that :math:`Y_i=Y_i(t)`.
* **A1(b) (Unconfoundedness)** :math:`Y_i(t)` is conditionally independent of :math:`T` given :math:`\textbf{S}`.
* **A1(c) (Treatment Variation)** The conditional variance of :math:`T` given any :math:`\textbf{S}=\textbf{s}` is strictly positive, i.e., :math:`\text{Var}(T|\textbf{S}=\textbf{s})>0`.
* **(A2) (Positivity)** The conditional density :math:`p(t|\textbf{s})` is bounded above and away from zero almost surely for all :math:`t` and :math:`\textbf{s}`.

There are three main estimation strategies for :math:`t\mapsto m(t)=\mathbb{E}\left[Y(t)\right]` and :math:`t\mapsto \theta(t)=\frac{d}{dt}\mathbb{E}\left[Y(t)\right]` with observed data :math:`\left\{(Y_i,T_i,\textbf{S}_i)\right\}_{i=1}^n` listed as follows.

* **Regression Adjustment (RA) Estimators:** 

.. math::

    \hat{m}_{RA}(t)  = \frac{1}{n}\sum_{i=1}^n \hat{\mu}(t,\textbf{S}_i),

where :math:`\hat{\mu}(t,\textbf{s})` is any consistent estimator of the conditional mean outcome function :math:`\mu(t,\textbf{s})=\mathbb{E}(Y|T=t,\textbf{S}=\textbf{s})`. Similarly,

.. math::

    \hat{\theta}_{RA}(t)  = \frac{1}{n}\sum_{i=1}^n \hat{\beta}(t,\textbf{S}_i),
    
where :math:`\hat{\beta}(t,\textbf{s})` is any consistent estimator of :math:`\beta(t,\textbf{s})=\frac{\partial}{\partial t}\mu(t,\textbf{s})`. 

* **Inverse Probability Weighting (IPW) Estimator:**

.. math::

    \hat{m}_{\mathrm{IPW}}(t) = \frac{1}{nh}\sum_{i=1}^n \frac{K\left(\frac{T_i-t}{h}\right)}{\hat{p}(T_i|\textbf{S}_i)}\cdot Y_i,
    
where :math:`h>0` is a smoothing bandwidth, :math:`K:\mathbb{R}\to [0,\infty)` is a kernel function, and :math:`\hat{p}(t|\textbf{s})` is a (consistent) estimator of the conditional density :math:`p(t|\textbf{s})`. Additionally,

.. math::

    \hat{\theta}_{\mathrm{IPW}}(t) = \frac{1}{nh^2}\sum_{i=1}^n \frac{Y_i\left(\frac{T_i-t}{h}\right)K\left(\frac{T_i-t}{h}\right)}{\kappa_2\cdot \hat{p}(T_i|\textbf{S}_i)},

where :math:`\kappa_2=\int u^2K(u)\,du>0`.

* **Doubly Robust (DR) Estimator:**

.. math::

    \hat{m}_{\mathrm{DR}}(t) =\frac{1}{nh}\sum_{i=1}^n \left\{\frac{K\left(\frac{T_i-t}{h}\right)}{\hat{p}(T_i|\textbf{S}_i)}\cdot \left[Y_i - \hat \mu(t,\textbf{S}_i)\right]+ h\cdot \hat{\mu}(t,\textbf{S}_i) \right\},

where :math:`\hat{\mu}(t,\textbf{s})` and :math:`\hat{p}(t,\textbf{s})` are (consistent) estimators of :math:`\mu(t,\textbf{s})` and :math:`p(t,\textbf{s})` respectively. The doubly robust estimator of :math:`\theta(t)` contains some new insights. For the outcome model, we need to specify and estimate both the condition mean outcome function :math:`\mu(t,\textbf{s})` and its partial derivative :math:`\beta(t,\textbf{s})` with respect to :math:`t` in order to obtain the following doubly robust estimator

.. math::

    \hat{\theta}_{\mathrm{DR}}(t) = \frac{1}{nh}\sum_{i=1}^n \left\{ \frac{\left(\frac{T_i-t}{h}\right)K\left(\frac{T_i-t}{h}\right) }{h\cdot \kappa_2\cdot \hat{p}(T_i|\textbf{S}_i)} \left[Y_i - \hat{\mu}(t,\textbf{S}_i) - (T_i-t)\cdot \hat{\beta}(t,\textbf{S}_i)\right]+ h\cdot \hat{\beta}(t,\textbf{S}_i) \right\}.




References
----------

.. [1] Yikun Zhang, Yen-Chi Chen, Alexander Giessing (2024+). Nonparametric Inference on Dose-Response Curves Without the Positivity Condition. *arXiv:2405.09003*

.. [2] Yikun Zhang, Yen-Chi Chen (2025+) Doubly Robust Inference on Causal Derivative Effects for Continuous Treatments. *arXiv:2501.06969*
