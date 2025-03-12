# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# Annotations and typing
from __future__ import annotations
from typing import List, Type, Any, Set, Optional, Callable

# Standard modules
import operator
from abc import ABC, abstractmethod
import warnings

# External modules
from scipy.stats import bernoulli, norm
import numpy as np
import dowhy.gcm as gcm


########################################################################################
# Abstract base classes


class CausalModel(gcm.StochasticModel, ABC):
    """Represents a causal model, i.e., a (potentially conditional) random variable.

    Causal models can be classified along multiple dimensions:
    - Parents: Unconditional (i.e., no parents) or conditional
    - Noise: Deterministic (i.e., no noise) or non-deterministic
    - Dimension: Univariate (i.e., one-dimensional) or multivariate
    - Domain: Continuous, discrete (with infinite domain), or finite

    There are abstract base classes for each for these model classes, indicated by the
    suffix _CausalModel_. Morevoer, there are base classes for more specific properties,
    like _InvertibleNoiseCausalModel_ or _BinaryCausalModel_.

    An instantiable class should inherit from all suitable base classes and should have
    the suffix _Model_. Example: _BernoulliModel_ inherits from
    InvertibleNoiseCausalModel, BinaryCausalModel, FiniteNoiseCausalModel, and
    UnivariateCausalModel.
    """

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


class ConditionalCausalModel(CausalModel, gcm.ConditionalStochasticModel, ABC):
    """Represents a causal model with parents."""

    def __init__(self, *, parent_signature: List[Type]):
        """
        :param parent_signature: An ordered list with the types of the model's parents.
        This can be used to check if a causal network is valid, i.e., if the
        predecessors of a node satisfy the type constraints of the model's parents.
        """
        self.parent_signature = parent_signature

    def __repr__(self):
        return f"<{self.__class__.__name__} parent_signature={self.parent_signature}>"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def draw_samples(self, parent_samples: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def probability(
        self, value: Any, *, parent_values: np.ndarray, comparison: str = operator.eq
    ) -> float:
        raise NotImplementedError


class UnconditionalCausalModel(CausalModel, ABC):
    """Represents a causal model without parents."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def draw_samples(self, num_samples: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def probability(self, value: Any, *, comparison: str = operator.eq) -> float:
        raise NotImplementedError


class DeterministicCausalModel(CausalModel, ABC):
    """Represents a causal model without noise."""
    pass


class NondeterministicCausalModel(CausalModel, ABC):
    """Represents a causal model with noise."""

    @abstractmethod
    def draw_noise_samples(self, num_samples: int) -> np.ndarray:
        raise NotImplementedError


# There are three fundamental classes of causal models:
# - Nondeterministic, no parents (NoiseCausalModel)
# - Deterministic, with parents (FunctionCausalModel)
# - Nondeterministic, with parents (NoiseFunctionCausalModel)
#
# The fourth class (deterministic, no parents) does not really add value and is
# therefore omitted.


class NoiseCausalModel(NondeterministicCausalModel, UnconditionalCausalModel, ABC):
    """Represents a causal model without parents, but with noise."""

    def draw_samples(self, num_samples: int) -> np.ndarray:
        return self.evaluate(self.draw_noise_samples(num_samples))

    def evaluate(self, noise_samples: np.ndarray) -> np.ndarray:
        return noise_samples


class FunctionCausalModel(DeterministicCausalModel, ConditionalCausalModel, ABC):
    """Represents a causal model with parents, but without noise."""

    @abstractmethod
    def evaluate(self, parent_samples: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NoiseFunctionCausalModel(
    NondeterministicCausalModel, ConditionalCausalModel, ABC
):
    """Represents a causal model with parents and noise."""

    def draw_samples(self, parent_samples: np.ndarray) -> np.ndarray:
        return self.evaluate(
            parent_samples, self.draw_noise_samples(len(parent_samples))
        )

    @abstractmethod
    def evaluate(
        self, parent_samples: np.ndarray, noise_samples: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError


class InvertibleCausalModel(NondeterministicCausalModel, ABC):
    pass


class InvertibleNoiseCausalModel(InvertibleCausalModel, NoiseCausalModel, ABC):
    """Represents a causal model without parents, but with noise, where the noise can
    be recovered from the values."""

    @abstractmethod
    def recover_noise(self, samples: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class InvertibleNoiseFunctionCausalModel(
    InvertibleCausalModel, NoiseFunctionCausalModel, ABC
):
    """Represents a causal model with parents and noise, where the noise can be
    recovered from the values."""

    @abstractmethod
    def recover_noise(
        self, samples: np.ndarray, parent_samples: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError


class UnivariateCausalModel(CausalModel, ABC):
    pass


class MultivariateCausalModel(CausalModel, ABC):
    def __init__(self, *, dim: int):
        self.dim = dim

    def __repr__(self):
        return f"<{self.__class__.__name__} dim={self.dim}>"


class ContinuousCausalModel(CausalModel, ABC):
    """Represents a causal model with continuous domain."""
    pass


class DiscreteCausalModel(CausalModel, ABC):
    """Represents a causal model with discrete domain."""
    pass


class FiniteCausalModel(DiscreteCausalModel, ABC):
    """Represents a causal model with finite domain."""

    def __init__(self, *, domain: Set[Any]):
        self.domain = domain

        DiscreteCausalModel.__init__(self)

    def __repr__(self):
        return f"<{self.__class__.__name__} domain={self.domain}>"


class BinaryCausalModel(FiniteCausalModel, ABC):
    """Represents a causal model with binary domain."""

    def __init__(self):
        FiniteCausalModel.__init__(self, domain={"0", "1"})


class FiniteNoiseCausalModel(NondeterministicCausalModel, FiniteCausalModel, ABC):
    """Represents a causal model with finite noise domain."""

    @abstractmethod
    def noise_support(self):
        raise NotImplementedError

    @abstractmethod
    def noise_probability(self, noise_value):
        raise NotImplementedError


########################################################################################
# Instantiable classes


class BernoulliModel(
    InvertibleNoiseCausalModel,
    BinaryCausalModel,
    FiniteNoiseCausalModel,
    UnivariateCausalModel,
):
    """Represents a Bernoulli variable."""

    def __init__(self, *, p: Optional[float] = None):
        self.p = p

        UnconditionalCausalModel.__init__(self)
        BinaryCausalModel.__init__(self)

    def __repr__(self):
        return f"<{self.__class__.__name__} p={self.p}>"

    def fit(self, X: np.ndarray) -> None:
        self.p = X.astype(int).mean()

    def draw_noise_samples(self, num_samples: int) -> np.ndarray:
        return bernoulli.rvs(p=self.p, size=num_samples).astype(str)

    def probability(self, value: Any, *, comparison: str = operator.eq) -> float:
        assert value in {"0", "1"}

        if comparison == operator.le:
            return 1.0 - self.p if value == "0" else 1.0
        elif comparison == operator.eq:
            return 1.0 - self.p if value == "0" else self.p
        elif comparison == operator.ge:
            return 1.0 if value == "0" else 1.0
        else:
            raise ValueError("Invalid operator!")

    def noise_support(self):
        return {"0", "1"}

    def noise_probability(self, noise_value):
        if noise_value == "0":
            return 1.0 - self.p
        elif noise_value == "1":
            return self.p
        else:
            raise ValueError(f"{noise_value} is no a valid noise value for this model!")

    def recover_noise(self, samples: np.ndarray) -> np.ndarray:
        return samples

    def clone(self):
        return BernoulliModel(p=self.p)


class NormalModel(
    InvertibleNoiseCausalModel, ContinuousCausalModel, UnivariateCausalModel
):
    """Represents a normal variable."""

    def __init__(self, *, loc: Optional[float] = None, scale: Optional[float] = None):
        self.loc = loc
        self.scale = scale

    def __repr__(self):
        return f"<{self.__class__.__name__} loc={self.loc}, scale={self.scale}>"

    def fit(self, X: np.ndarray) -> None:
        self.loc = X.mean()
        self.scale = X.std()

    def draw_noise_samples(self, num_samples: int) -> np.ndarray:
        return norm.rvs(loc=self.loc, scale=self.scale, size=num_samples)

    def probability(self, value: Any, *, comparison: str = operator.eq) -> float:
        if comparison == operator.le:
            return norm.cdf(value)
        elif comparison == operator.eq:
            warnings.warn(
                "The probability of equality is always zero for continuous \
                    distributions!"
            )
            return 0.0
        elif comparison == operator.ge:
            return 1.0 - norm.cdf(value)
        else:
            raise ValueError("Invalid operator!")

    def recover_noise(self, samples: np.ndarray) -> np.ndarray:
        return samples

    def clone(self):
        return NormalModel(loc=self.loc, scale=self.scale)


class FunctionModel(FunctionCausalModel):
    """Represents a random variable which is a deterministic function of its parents."""

    def __init__(self, *, parent_signature: List[Type], f: Callable):
        self.f = f

        ConditionalCausalModel.__init__(self, parent_signature=parent_signature)

    def __repr__(self):
        return f"<{self.__class__.__name__} parent_signature={self.parent_signature}, \
              f={self.f}>"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        pass

    def draw_samples(self, parent_samples: np.ndarray) -> np.ndarray:
        return self.f(parent_samples)

    def evaluate(self, parent_samples: np.ndarray) -> np.ndarray:
        return self.f(parent_samples)

    def probability(
        self, value: Any, *, parent_values: np.ndarray, comparison: str = operator.eq
    ) -> float:
        return 1.0 if value == self.f(parent_values) else 0.0

    def clone(self):
        return FunctionModel(parent_signature=self.parent_signature, f=self.f)


class BinaryFunctionModel(FunctionModel, BinaryCausalModel):
    """Represents a binary random variable which is a deterministic function of its
      parents."""

    def __init__(self, *, parent_signature: List[Type], f: Callable):
        FunctionModel.__init__(self, parent_signature=parent_signature, f=f)
        BinaryCausalModel.__init__(self)
