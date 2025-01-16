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

    \hat{m}_{\mathrm{IPW}}(t) = \frac{1}{nh}\sum_{i=1}^n \frac{K\left(\frac{T_i-t}{h}\right)}{\hat{p}_{T|\textbf{S}}(T_i|\textbf{S}_i)}\cdot Y_i,
    
where :math:`h>0` is a smoothing bandwidth, :math:`K:\mathbb{R}\to [0,\infty)` is a kernel function, and :math:`\hat{p}(t|\textbf{s})` is a (consistent) estimator of the conditional density :math:`p(t|\textbf{s})`. Additionally,

.. math::

    \hat{\theta}_{\mathrm{IPW}}(t) = \frac{1}{nh^2}\sum_{i=1}^n \frac{Y_i\left(\frac{T_i-t}{h}\right)K\left(\frac{T_i-t}{h}\right)}{\kappa_2\cdot \hat{p}_{T|\textbf{S}}(T_i|\textbf{S}_i)},

where :math:`\kappa_2=\int u^2K(u)\,du>0`.

* **Doubly Robust (DR) Estimator:**

.. math::

    \hat{m}_{\mathrm{DR}}(t) =\frac{1}{nh}\sum_{i=1}^n \left\{\frac{K\left(\frac{T_i-t}{h}\right)}{\hat{p}_{T|\textbf{S}}(T_i|\textbf{S}_i)}\cdot \left[Y_i - \hat \mu(t,\textbf{S}_i)\right]+ h\cdot \hat{\mu}(t,\textbf{S}_i) \right\},

where :math:`\hat{\mu}(t,\textbf{s})` and :math:`\hat{p}(t,\textbf{s})` are (consistent) estimators of :math:`\mu(t,\textbf{s})` and :math:`p(t,\textbf{s})` respectively. The doubly robust estimator of :math:`\theta(t)` contains some new insights. For the outcome model, we need to specify and estimate both the condition mean outcome function :math:`\mu(t,\textbf{s})` and its partial derivative :math:`\beta(t,\textbf{s})` with respect to :math:`t` in order to obtain the following doubly robust estimator

.. math::

    \hat{\theta}_{\mathrm{DR}}(t) = \frac{1}{nh}\sum_{i=1}^n \left\{ \frac{\left(\frac{T_i-t}{h}\right)K\left(\frac{T_i-t}{h}\right) }{h\cdot \kappa_2\cdot \hat{p}_{T|\textbf{S}}(T_i|\textbf{S}_i)} \left[Y_i - \hat{\mu}(t,\textbf{S}_i) - (T_i-t)\cdot \hat{\beta}(t,\textbf{S}_i)\right]+ h\cdot \hat{\beta}(t,\textbf{S}_i) \right\}.

Furthermore, 

.. math::

    \sqrt{nh^3}\left[\hat{\theta}_{\mathrm{DR}}(t) - \theta(t) - h^2 B_{\theta}(t)\right] \stackrel{d}{\to} \mathcal{N}\left(0,V_{\theta}(t)\right),
    
where :math:`B_{\theta}(t)` is a bias term. By choosing a bandwidth with a standard rate of convergence :math:`h=O\left(n^{-1/5}\right)`, we can construct a :math:`(1-\alpha)`-level confidence interval for :math:`\theta(t)` as:
 
 .. math::
     
     \left[\hat{\theta}_{\mathrm{DR}}(t)- \Phi\left(1-\frac{\alpha}{2}\right)\sqrt{\frac{\hat{V}_{\theta}(t)}{nh^3}},\; \hat{\theta}_{\mathrm{DR}}(t)+ \Phi\left(1-\frac{\alpha}{2}\right)\sqrt{\frac{\hat{V}_{\theta}(t)}{nh^3}}\right],

where :math:`\Phi(\cdot)` is the cumulative distribution function of :math:`\mathcal{N}(0,1)` and :math:`\hat{V}_{\theta}(t)` is computed as:

 .. math::
 
     \hat{V}_{\theta}(t) = \frac{1}{n} \sum_{i=1}^n \left\{\phi_{h,t}\left(Y_i,T_i,\textbf{S}_i;\hat{\mu}, \hat{\beta}, \hat{p}_{T|\textbf{S}}\right) + \sqrt{h^3}\left[\hat{\beta}(t,\textbf{S}_i) - \hat{\theta}_{\mathrm{DR}}(t) \right]\right\}^2

with :math:`\phi_{h,t}\left(Y,T,\textbf{S}; \bar{\mu},\bar{\beta}, \bar{p}_{T|\textbf{S}}\right) = \frac{\left(\frac{T-t}{h}\right) K\left(\frac{T-t}{h}\right)}{\sqrt{h}\cdot \kappa_2\cdot \bar{p}_{T|\textbf{S}}(T|\textbf{S})}\cdot \left[Y - \bar{\mu}(t,\textbf{S}) - (T-t)\cdot \bar{\beta}(t,\textbf{S})\right]`.


Identification and Estimation Without Positivity
------------

To study the IPW and DR estimators without relying on the positivity condition, we impose an additive structural assumption on the potential outcome as :math:`Y(t) = \bar{m}(t) + \eta(\textbf{S}) +\epsilon`. The identification theory in Section 2 of [1]_ implies that both the dose-response curve :math:`m(t)` and its derivative :math:`\theta(t)` are identifiable even under violations of positivity. However, the aforementioned IPW and DR estimators are indeed biased without positivity even when the additive structural assumption holds true; see Section 4.2 in [2]_.

We propose the following bias-corrected IPW and DR estimators of :math:`\theta(t)` as:

 .. math::
 
     \hat{\theta}_{\mathrm{C,IPW}}(t) = \frac{1}{nh^2} \sum_{i=1}^n \frac{Y_i\left(\frac{T_i-t}{h}\right) K\left(\frac{T_i-t}{h}\right) \hat{p}_{\zeta}(\textbf{S}_i|t)}{\kappa_2 \cdot \hat{p}(T_i,\textbf{S}_i)}
     
     \hat{\theta}_{\mathrm{C,DR}}(t) = \frac{1}{nh^2} \sum_{i=1}^n \frac{\left(\frac{T_i-t}{h}\right) K\left(\frac{T_i-t}{h}\right) \hat{p}_{\zeta}(\textbf{S}_i|t)}{\kappa_2\cdot \hat{p}(T_i,\textbf{S}_i)} \left[Y_i - \hat{\mu}(t,\textbf{S}_i) - (T_i-t)\cdot \hat{\beta}(t,\textbf{S}_i)\right] + \int \hat{\beta}(t,\textbf{s})\cdot \hat{p}_{\zeta}(\textbf{s}|t)\, d\textbf{s},

where :math:`\hat{p}(t,\textbf{s})` is a consistent estimator of the joint density :math:`p(t,\textbf{s})` and :math:`\hat{p}_{\zeta}(\textbf{s}|t)` is an estimated :math:`\zeta`-interior conditional density defined as:

.. math::

    p_{\zeta}(\textbf{s}|t) = \frac{p_{\textbf{S}|T}(\textbf{s}|t) \cdot \mathbb{1}_{\left\{\textbf{s}\in \mathcal{L}_{\zeta}(t)\right\}}}{\int_{\mathcal{L}_{\zeta}(t)} p_{\textbf{S}|T}(\textbf{s}_1|t) \,d\textbf{s}_1},
    
with :math:`\mathcal{L}_{\zeta}(t) = \left\{\textbf{s}\in \mathcal{S}(t): p_{\textbf{S}|T}(\textbf{s}|t) \geq \zeta\right\}` being the :math:`\zeta`-upper level set of the conditional density :math:`p_{\textbf{S}|T}(\textbf{s}|t)`.

It can be proved that

.. math::

    \sqrt{nh^3}\left[\hat{\theta}_{\mathrm{C,DR}}(t) - \theta(t) - h^2 B_{C,\theta}(t)\right] \stackrel{d}{\to} \mathcal{N}\left(0,V_{C,\theta}(t)\right),
    
where :math:`B_{C,\theta}(t)` is a bias term. By choosing a bandwidth with a standard rate of convergence :math:`h=O\left(n^{-1/5}\right)`, we can construct a :math:`(1-\alpha)`-level confidence interval for :math:`\theta(t)` as before, in which the asymptotic variance is estimated by

.. math::

    \hat{V}_{C,\theta}(t) = \frac{1}{n} \sum_{i=1}^n \left\{\phi_{C,h,t}\left(Y_i,T_i,\textbf{S}_i;\hat{\mu}, \hat{\beta}, \hat{p}, \hat{p}_{\zeta}\right) + \sqrt{h^3}\left[\int \hat{\beta}(t,\textbf{s}) \cdot \hat{p}_{\zeta}(\textbf{s}|t)\, d\textbf{s} - \hat{\theta}_{\mathrm{C,DR}}(t) \right]\right\}^2,
    
where :math:`\phi_{C,h,t}\left(Y,T,\textbf{S}; \bar{\mu},\bar{\beta}, \bar{p},\bar{p}_{\zeta}\right) = \frac{\left(\frac{T-t}{h}\right) K\left(\frac{T-t}{h}\right) \cdot \bar{p}_{\zeta}(\textbf{S}|t)}{\sqrt{h}\cdot \kappa_2\cdot \bar{p}(T,\textbf{S})}\cdot \left[Y - \bar{\mu}(t,\textbf{S}) - (T-t)\cdot \bar{\beta}(t,\textbf{S})\right]`.

References
----------

.. [1] Yikun Zhang, Yen-Chi Chen, Alexander Giessing (2024+). Nonparametric Inference on Dose-Response Curves Without the Positivity Condition. *arXiv:2405.09003*

.. [2] Yikun Zhang, Yen-Chi Chen (2025+) Doubly Robust Inference on Causal Derivative Effects for Continuous Treatments. *arXiv:2501.06969*
