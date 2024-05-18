[![PyPI pyversions](https://img.shields.io/pypi/pyversions/npDoseResponse.svg)](https://pypi.python.org/pypi/npDoseResponse/)
[![PyPI version](https://badge.fury.io/py/npDoseResponse.svg)](https://badge.fury.io/py/NPDoseResponse)
[![Downloads](https://static.pepy.tech/badge/npDoseResponse)](https://pepy.tech/project/npDoseResponse)
[![Documentation Status](https://readthedocs.org/projects/npdoseresponse/badge/?version=latest)](http://npdoseresponse.readthedocs.io/?badge=latest)

# Nonparametric Estimation and Inference on Dose-Response Curves

This package implements the proposed integral estimator and a localized derivative estimator for estimating the covariate-adjusted regression function (or the dose-response curve in causal inference) and its derivative. We also document all the code for simulations and real-world case study in our paper [here](https://github.com/zhangyk8/NPDoseResponse/tree/main/Paper_Code).

* Free software: MIT license
* Python Package Documentation: [https://npdoseresponse.readthedocs.io](https://npdoseresponse.readthedocs.io).

Installation guide
--------

```npDoseResponse``` requires Python 3.8+ (earlier version might be applicable) and [NumPy](http://www.numpy.org/). To install the latest version of ```npDoseResponse``` from this repository, run:

```
python setup.py install
```

To pip install a stable release, run:
```
pip install npDoseResponse
```

References
--------

<a name="npdoseresponse">[1]</a> Y. Zhang, Y.-C. Chen, and A. Giessing (2024+) Nonparametric Inference on Dose-Response Curves Without the Positivity Condition [arXiv:2405.09003](https://arxiv.org/abs/2405.09003).

