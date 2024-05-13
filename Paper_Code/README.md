## Simulation Studies

We consider three different model settings for our simulation studies. For each model setting, we generate <img src="https://latex.codecogs.com/svg.latex?&space;\left\{(Y_i,T_i,\textbf{S}_i)\right\}_{i=1}^n"/> with <img src="https://latex.codecogs.com/svg.latex?&space;n=2000"/> independent and identically distributed observations, where <img src="https://latex.codecogs.com/svg.latex?&space;Y\in\mathbb{R}"/> is the outcome variable, <img src="https://latex.codecogs.com/svg.latex?&space;T\in\mathbb{R}"/> is the continuous treatment variable, and <img src="https://latex.codecogs.com/svg.latex?&space;\textbf{S}\in\mathbb{R}^d"/> is a covariate vector of confounding variables.

- **Single Confounding Model:**
<img src="https://latex.codecogs.com/svg.latex?\large&space;Y=T^2+T+1+10S+\epsilon,\quad\,T=\sin(\pi\,S)+E,\quad\,S\sim\text{Uniform}[-1,1]\subset\mathbb{R},\quad\,E\sim\text{Uniform}[-1,1],\quad\text{and}\quad\epsilon\sim\mathcal{N}(0,1)."/>


<p align="center">
<img src="https://github.com/zhangyk8/NPDoseResponse/blob/main/Paper_Code/Figures/single_conf_TS.png" style="zoom:50%" />
 <br><B>Fig 1. </B>The support of the joint distribution of <img src="https://latex.codecogs.com/svg.latex?&space;(T,S)"/>.
 </p>

## Case Study: Effect of PM2.5 on Cardiovascular Mortality Rate
