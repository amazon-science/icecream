# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from abc import ABC
from typing import List, Type, Any
import operator
import math

import numpy as np
from scipy.stats import bernoulli
import pandas as pd

import dowhy.gcm as gcm

from explain import ConditionalCausalModel, BinaryCausalModel, FiniteNoiseCausalModel, NoiseFunctionCausalModel, \
    NoiseCausalModel


class CloudServiceErrorModel(BinaryCausalModel, FiniteNoiseCausalModel, ABC):
    """Represents a general causal model for a node of a cloud computing application. Such a node can produce an error
    (with probability p), and in case it has dependencies (parents), it can also propagate errors from them.
    """
    def __init__(self, p: float):
        self.p = p

        BinaryCausalModel.__init__(self)

    def noise_support(self):
        return {'0', '1'}

    def clone(self):
        pass

    def __repr__(self):
        return f'<{self.__class__.__name__} domain={self.domain}, p={self.p:.2}>'

    def noise_probability(self, noise_value):
        if noise_value == '0':
            return 1.0 - self.p
        elif noise_value == '1':
            return self.p
        else:
            raise ValueError(f'{noise_value} is no a valid noise value for this model!')


class CloudServiceErrorConditionalModel(NoiseFunctionCausalModel, CloudServiceErrorModel,
                                        gcm.graph.InvertibleFunctionalCausalModel):
    """Represents a non-root node in a cloud computing application. An error occurs when one is produced at the node, or
    when the number of parent errors reaches the threshold t."""
    def __init__(self, *, parent_signature: List[Type], t: int, p: float):
        self.t = t
        self.p = p

        ConditionalCausalModel.__init__(self, parent_signature=parent_signature)
        CloudServiceErrorModel.__init__(self, p=p)

    def draw_noise_samples(self, num_samples: int) -> np.ndarray:
        return bernoulli.rvs(size=num_samples, p=self.p).astype(str)

    def evaluate(self, parent_samples: np.ndarray, noise_samples: np.ndarray) -> np.ndarray:
        return np.maximum(parent_samples.astype(int).sum(axis=1) >= self.t, noise_samples.astype(int)).astype(str)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        data = pd.DataFrame({'sum': X.astype(int).sum(axis=1), 'target': Y.astype(int)})
        groups = data.groupby('sum').mean()
        threshold = groups[groups['target'] == 1].index.min()

        if math.isnan(threshold):
            self.t = X.shape[1] + 1
            self.p = data['target'].mean()
        else:
            self.t = threshold
            self.p = data['target'][data['sum'] < self.t].mean()

    def probability(self, value: Any, *, parent_values: np.ndarray, comparison: str = operator.eq) -> float:
        raise NotImplementedError

    def estimate_noise(self, target_samples: np.ndarray, parent_samples: np.ndarray) -> np.ndarray:
        return (target_samples.astype(int) * np.maximum(parent_samples.astype(int).sum(axis=1) < self.t,
                                                        bernoulli.rvs(size=len(target_samples), p=self.p))).astype(str)

    def __repr__(self):
        return f'<{self.__class__.__name__} domain={self.domain}, p={self.p:.2}, ' \
               f't={self.t}/{len(self.parent_signature)}>'


class CloudServiceErrorRootModel(NoiseCausalModel, CloudServiceErrorModel, gcm.graph.StochasticModel):
    """Represents a root node in a cloud computing application."""
    def __init__(self, *, p: float):
        CloudServiceErrorModel.__init__(self, p=p)

    def draw_samples(self, num_samples: int) -> np.ndarray:
        return self.draw_noise_samples(num_samples)

    def draw_noise_samples(self, num_samples: int) -> np.ndarray:
        return bernoulli.rvs(size=num_samples, p=self.p).astype(str)

    def fit(self, X: np.ndarray) -> None:
        self.p = X.astype(int).mean()

    def probability(self, value: Any, *, comparison: str = operator.eq) -> float:
        raise NotImplementedError
