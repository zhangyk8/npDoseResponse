## Simulation Studies

We consider three different model settings for our simulation studies. For each model setting, we generate <img src="https://latex.codecogs.com/svg.latex?&space;\left\{(Y_i,T_i,\textbf{S}_i)\right\}_{i=1}^n"/> with <img src="https://latex.codecogs.com/svg.latex?&space;n=2000"/> independent and identically distributed observations, where <img src="https://latex.codecogs.com/svg.latex?&space;Y\in\mathbb{R}"/> is the outcome variable, <img src="https://latex.codecogs.com/svg.latex?&space;T\in\mathbb{R}"/> is the continuous treatment variable, and <img src="https://latex.codecogs.com/svg.latex?&space;\textbf{S}\in\mathbb{R}^d"/> is a covariate vector of confounding variables.

- **Single Confounder Model:**
<img src="https://latex.codecogs.com/svg.latex?\large&space;Y=T^2+T+1+10S+\epsilon,\quad\,T=\sin(\pi\,S)+E,\quad\,S\sim\text{Uniform}[-1,1]\subset\mathbb{R},\quad\,E\sim\text{Uniform}[-1,1],\quad\text{and}\quad\epsilon\sim\mathcal{N}(0,1)."/>

<p align="center">
<img src="https://github.com/zhangyk8/NPDoseResponse/blob/main/Paper_Code/Figures/single_conf_TS.png" style="zoom:50%" />
 <br><B>Fig 1. </B>The support of the joint distribution of <img src="https://latex.codecogs.com/svg.latex?&space;(T,S)"/>.
 </p>

One can easily see that the conditional density <img src="https://latex.codecogs.com/svg.latex?&space;p(t|s)"/> is 0 within the gray regions of **Fig 1**, suggesting that the positivity condition fails. The simulation on this single confounder model is conducted in the Python script `Single_Confounder_Model.py`.

- **Linear Confounding Model:**
<img src="https://latex.codecogs.com/svg.latex?\large&space;Y=T+6S_1+6S_2+\epsilon,\quad\,T=2S_1+S_2+E,\quad\,\textbf{S}=(S_1,S_2)\sim\text{Uniform}[-1,1]\subset\mathbb{R},\quad\,E\sim\text{Uniform}[-1,1]^2,\quad\text{and}\quad\epsilon\sim\mathcal{N}(0,1)."/>

The simulation on this linear confounding model is conducted in the Python script `Linear_Confounding_Model.py`.

- **Nonlinear Confounding Model:**
<img src="https://latex.codecogs.com/svg.latex?\large&space;Y=T^2+T+10Z+\epsilon,\quad\,T=\cos\left(\pi\,Z^3\right)+\frac{Z}{4}+E,\quad\,Z=4S_1+S_2,\quad\textbf{S}=(S_1,S_2)\sim\text{Uniform}[-1,1]\subset\mathbb{R},\quad\,E\sim\text{Uniform}[-1,1]^2,\quad\text{and}\quad\epsilon\sim\mathcal{N}(0,1)."/>

The simulation on this linear confounding model is conducted in the Python script `Nonlinear_Effect_Model1.py`.

## Case Study: Effect of PM2.5 on Cardiovascular Mortality Rate (CMR)

We apply our proposed estimators to analyzing the relationship between the PM2.5 level and CMR from 1990 to 2010 in 2132 counties of the United States. The data are obtained from Wyatt et al. (2020), and the code is in the folder `PM25_App`.

- Data preprocessing: `PM25_DataPreprocess.py`.
- Local quadratic regression of Y on T only: `PM25_App_Tonly.py`.
- Our proposed estimators of regressing Y on T and spatial locations: `PM25_App_TS.py`.
- Our proposed estimators of regressing Y on T and all the covariates (spatial locations + socioeconomic factors): `PM25_App_Full.py`.

## References

<a name="npdoseresponse">[1]</a> Y. Zhang, Y.-C. Chen, A. Giessing (2024+) Nonparametric Inference on Dose-Response Curves Without the Positivity Condition.

<a name="data">[2]</a> L. Wyatt, G. Peterson, T. Wade, L. Neas, and A. Rappold (2020). Annual PM2.5 and cardiovascular mortality rate data: Trends modified by county socioeconomic status in 2,132 US counties. _Data in Brief_ **30** 105--318.

