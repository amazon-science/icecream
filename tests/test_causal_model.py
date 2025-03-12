# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

from explain import (
    CausalModel,
    NondeterministicCausalModel,
    UnconditionalCausalModel,
    BinaryCausalModel,
    FiniteNoiseCausalModel,
    BernoulliModel,
)


def test_given_bernoulli_model_then_isinstance_returns_expected_result():
    causal_model = BernoulliModel(p=0.5)

    assert isinstance(causal_model, CausalModel)
    assert isinstance(causal_model, NondeterministicCausalModel)
    assert isinstance(causal_model, UnconditionalCausalModel)
    assert isinstance(causal_model, BinaryCausalModel)
    assert isinstance(causal_model, FiniteNoiseCausalModel)


def test_given_bernoulli_model_then_sample_returns_expected_result():
    causal_model = BernoulliModel(p=0.5)

    assert causal_model.draw_samples(10).shape == (10,)
    assert causal_model.draw_samples(10).dtype.type == np.str_


def test_given_bernoulli_model_then_probability_returns_expected_result():
    causal_model = BernoulliModel(p=0.6)

    assert causal_model.probability('0') == 0.4
    assert causal_model.probability('1') == 0.6

    with pytest.raises(AssertionError):
        causal_model.probability('2')
