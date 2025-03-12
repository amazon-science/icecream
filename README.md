# ICECREAM: Identifying Coalition-based Explanations for Common and Rare Events in Any Model

## Citation

Oesterle, M., Blöbaum, P., Mastakouri, A. A., & Kirschbaum, E. (2025). Beyond Single-Feature Importance with ICECREAM. In _Causal Learning and Reasoning (CLeaR)_. PMLR.

## Repository structure
### `/src/`
The explanation score and related functionalities are provided by the module `src.explain`.

### `/test/`
Contains test functions for the `src.explain` module.

### `/experiments/`
Contains all experiments that are part of the paper (see also [this section](#experiments-1)). For each experiment, there is a notebook containing the actual experiment. Additionally, some experiments include specific causal models or other functionality, as well as data.


## Functionality
The theory behind this package is described in the paper "Oesterle et al. (2025). _Beyond Single-Feature Importance with ICECREAM_" which can be found at https://arxiv.org/abs/XXX.

The fundamental idea is a quantitative measure of "explanatory power" which describes how well a coalition of nodes explains
the value of the target node for a specific observation. This measure (called _explanation score_) can be either 
calculated from a causal model (see `src.explain.causal_explanation`), from a classifier model (see `src.explain.model_explanation`), 
or estimated purely from data samples  (see `src.explain.data_explanation`). It is defined as the relative distance of certain
probability distributions, using the folowing concepts:

### Grounding
For a random variable `Y` which is a function of features `X_1` ..., `X_n`, i.e., `Y = f(X_1, ..., X_n)` and an 
observation `x = (x_1, ..., x_n)` , a _grounding_ of `Y` with respect to the coalition `C \subseteq I = {1, ..., n}` is 
defined as the interventional variable `Y|_C := Y|do(X_C = x_C)`. For `C = \varnothing`, `Y|_C = Y`, while for 
`C = I`, `Y|_C = \delta_{f(x)}`.

### Distance measure
For probability distributions `P`, `Q` over the same domain, we use the KL divergence `D_KL(Q || P)` as the distance
`d(P, Q)`.

### Explanation Score
The explanation score of a coalition `C` for an observation `x` is defined as 

```
E_x(C) := (d(Y, Y_I) - d(Y_C, Y_I)) / d(Y, Y_I)
```


## Experiments
#### CorrAL (`/experiments/corral/`)
CorrAL is a synthetic dataset consisting of six Boolean features (`A0`, `A1`, `B0`, `B1`, `Irrelevant` and `Correlated`) and 160 samples. The feature names are rather self-explanatory as the target variable is defined as
`class := (A0 and A1) or (B0 and B1)`.
The remaining two features are not used--`Irrelevant` is uniformly random, and `Correlated` matches the target label in 75% of all samples. John et al. (1994), who created the dataset as an illustration for the shortcomings of existing feature selection algorithms, report that greedy selection mechanisms (both top-down and bottom-up) have difficulties identifying the correct feature set. We therefore test _ICECREAM_ on the full dataset and see which features are selected as part of the minimum-size coalitions with full explanation score.

#### South German Credit Dataset (`/experiments/credit/`)
The South German Credit dataset (Grömping, 2019) contains 1,000 loan applications with 21 features, and the corresponding risk assessment (low-risk or high-risk) as the binary target variable. We train a simple random forest classifier on the dataset whose outputs we want to explain. We apply _ICECREAM_ to the trained model, and compare the results with the instance-wise scores of the popular explainability framework SHAP (Lundberg and Lee, 2017).

#### Cloud Computing (`/experiments/cloud/`)
We consider a cloud computing application in which services call each other to produce a result which is returned to the user at the target service `Y`. Services can either be stand-alone (like a database or a random number generator) or call other services and transform their input to produce an output. When we observe an error at the target service `Y`, we want to know which service(s) caused this error. To imitate such a system, we synthetically generate data from a system with ten binary variables. Each variable represents a service with an intrinsic error rate and a threshold which determines how many parent services must experience an error such that the error is propagated.

We compare _ICECREAM_ to the root cause analysis (RCA) approaches presented by Budhathoki et al. (2022).
