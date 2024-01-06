# ARMUL: Adaptive and Robust Multi-Task Learning
Paper: Duan, Y. and Wang, K., 2023. Adaptive and robust multi-task learning. The Annals of Statistics, 51(5), pp.2015-2039. (https://arxiv.org/abs/2202.05250).

## Introduction

Suppose there are $m$ datasets (tasks), the $j$-th of which has $n_j$ samples and an empirical loss function $f_j:~\mathbb{R}^d \to \mathbb{R}$. Multi-Task Learning (MTL) with structural constraint is often formulated as

$$ \min_{\Theta \in \Omega} \bigg\lbrace \sum_{j=1}^m w_j f_j(\theta_j) \bigg\rbrace, $$

where $w_j$'s are weights, each column of $\Theta = (\theta_1,\cdots,\theta_m) \in \mathbb{R}^{d\times m}$ is the parameter vector of a task, and $\Omega  \subseteq \mathbb{R}^{d\times m}$ encodes the prior knowledge of task relatedness. To handle possible misspecification of the structure, we propose a method named Adaptive and Robust MUlti-Task Learning (ARMUL):

$$ \min_{ \Theta \in \mathbb{R}^{d\times m}, \Gamma \in \Omega} 
\bigg\lbrace \sum_{j=1}^m w_j [ f_j(\theta_j) +
\lambda_j \Vert \theta_j - \gamma_j \Vert_2 ]
\bigg\rbrace.$$

Denote by $(\widehat{\Theta}, \widehat{\Gamma})$ an optimal solution. The $\widehat{\Theta}$ above estimates parameters for the $m$ tasks, which is shrunk toward a prototype $\widehat{\Gamma}$ in the prescribed model space $\Omega$ so as to promote task relatedness. The $\lambda_j$'s are penalty parameters. Theories suggest taking $\lambda_j = C \sqrt{ d / n_j }$ for some constant $C$ shared by all tasks. The tuning parameter $C$ can be selected using cross-validation.

## Get Started

The Python classes `ARMUL` and `Baselines` in `ARMUL.py` implement three important cases of ARMUL (vanilla, clustered and low-rank) as well as four baseline procedures (single-task learning, data pooling, clustered MTL and low-rank MTL). The current version supports

1. multi-task linear regression: 

$$f_j(\theta) = \frac{1}{n_j} \sum_{i=1}^{n_j} (x_{ij}^{\top} \theta - y_{ij})^2,$$ 

2. multi-task logistic regression:

$$f_j(\theta) = \frac{1}{n_j} \sum_{i=1}^{n_j} [y_{ij} x_{ij}^{\top} \theta - \log(1 + e^{x_{ij}^{\top} \theta}) ],$$

where $y_{ij} \in \lbrace 0 , 1 \rbrace$. 


### ARMUL methods

Preparation: store the $m$ datasets using a list named `data` of length 2. `data[0]` is a list of $m$ feature matrices, the $j$-th of which is a 2-dimensional numpy array of size $(n_j, d)$. `data[1]` is a list of $m$ response vectors, the $j$-th of which is a 2-dimensional numpy array of size $(n_j, 1)$. Define a list named `lbd` of $m$ penalty parameters $(\lambda_1,\cdots,\lambda_m)$, not necessarily following the suggestion $\lambda_j = C \sqrt{d / n_j}$. 

To implement vanilla ARMUL, run
```sh
import numpy as np
from ARMUL import ARMUL
test = ARMUL(link) # link == 'linear' or 'logistic'
test.vanilla(data, lbd, standardization = True, intercept = True)
```

If `standardization = True` (default), the raw features (for linear and logistic regression) and responses (for linear regression only) will be standardized to have zero mean and unit variance. If `intercept = True` (default), an all-one feature will be added to the datasets. See `ARMUL.py` for other arguments of the function `vanilla`, such as the step size and number of iterations of the proximal gradient descent algorithm for optimization. The two components $\widehat{\Theta}$ and $\widehat{\Gamma}$ of the optimal solution can be retrieved from `test.models['vanilla']` and `test.models['vanilla_gamma']`, respectively. To evaluate the out-of-sample performance, prepare testing data (`data_test`) from the $m$ tasks in the same form as the training data (`data`). Then, execute
```sh
test.evaluate(data_test, model = 'vanilla')
```
This computes the testing errors (mean square errors for linear regression and misclassification errors for logistic regression) on the $m$ tasks. The $m$ individual errors are stored at `test.results['errors']`, whose average (weighted by sample sizes) is `test.results['average error']`. For linear regression, `test.results['R2']` returns the overall R-square on all the data.


Clustered and low-rank ARMUL can be implemented using `test.clustered(data, lbd, K)` and `test.lowrank(data, lbd, K)`, respectively. Here `K` is the number of clusters or the rank.

### Selecting tuning parameters by cross-validation

Vanilla ARMUL has tuning parameters $(\lambda_1,\cdots,\lambda_m)$. Suppose there are $S$ such configurations $\lbrace ( \lambda_{1}^{(s)}, \cdots, \lambda_{m}^{(s)} ) \rbrace_{s=1}^S$ to choose from. We define a list `lbd_list` of length $S$ with `lbd_list[s]` being the list $(\lambda_{1}^{(s-1)}, \cdots, \lambda_m^{(s-1)} )$ of length $m$. Then, running
```sh
test.cv(data, lbd_list, model = 'vanilla', n_fold = 5, seed = 1000, standardization = True, intercept = True)
```
estimates the testing error (mean square errors or misclassification errors averaged over all tasks, weighted by sample sizes) of vanilla ARMUL with each configuration $( \lambda_{1}^{(s)}, \cdots, \lambda_{m}^{(s)} )$ using 5-fold cross-validation. Here `n_fold` is the number of folds and `seed` is the random seed. After that,

1. `test.errors_cv` stores the validations errors corresponding to the $S$ candidate configurations;

2. `test.lbd_cv` is the selected configuration;

3. vanilla ARMUL is refitted on all the training data using the selected configuration, and `test.models['vanilla']` gives the final $\widehat{\Theta}$.

We can evaluate the obtained $\widehat{\Theta}$ on testing data by running `test.evaluate(data_test, model = 'vanilla')` as before. The above procedure is helpful for selecting the constant $C$ in the suggested expression $\lambda_j = C \sqrt{d / n_j}$. 

To apply cross-validation to clustered or low-rank ARMUL, set `model = 'clustered'` or  `model = 'lowrank'`, choose a value for `K`, and execute
```sh
test.cv(data, lbd_list, model = model, K = K, n_fold = 5, seed = 1000, standardization = True, intercept = True)
```
For a given `K`, it uses cross-validation to select from multiple configurations of $\lambda_j$'s in `lbd_list`.


### Baseline procedures


To implement baseline procedures, run

```sh
import numpy as np
from ARMUL import Baselines
base = Baselines(link) # link == 'linear' or 'logistic'
```

Single-task learning, data pooling, clustered MTL and low-rank MTL correspond to `base.STL_train(data)`, `base.DP_train(data)`, `base.clustered_train(data, K)` and `base.lowrank_train(data, K)`, respectively. See `ARMUL.py` for other arguments of those methods. The models can be evaluated using `base.evaluate(data_test, model)`, where `model` is `'STL'`, `'DP'`, `'clustered'` or `'lowrank'`.


## Demonstration

See `Demo.ipynb` for a real data example.


## Experiments in the paper

The folder `experiments in the paper` contains all the experimental results and codes of the ARMUL paper. See `Demo_reproducibility.ipynb` for how to reproduce the results.


## Citation
```
@article{DW23,
  title={Adaptive and robust multi-task learning},
  author={Duan, Yaqi and Wang, Kaizheng},
  journal={The Annals of Statistics},
  volume={51},
  number={5},
  pages={2015--2039},
  year={2023},
  publisher={Institute of Mathematical Statistics}
}
```
