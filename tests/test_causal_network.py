# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

from explain import CausalNetwork, BernoulliModel, BinaryFunctionModel


def test_given_causal_models_then_building_causal_network_returns_expected_result():
    Z = BernoulliModel(p=0.5)
    Y = BinaryFunctionModel(
        parent_signature=[BernoulliModel],
        f=lambda parent_values: (1 - parent_values.astype(int)).astype(str),
    )

    causal_network = CausalNetwork({'Y': (Y, ['Z']), 'Z': Z})

    assert list(causal_network.nodes) == ['Z', 'Y']
    with pytest.raises(ValueError):
        causal_network['X']

    assert causal_network.parents('Y') == ['Z']

    with pytest.raises(AssertionError):
        causal_network.set_causal_model('Z', Y)


def test_given_causal_network_then_draw_samples_returns_expected_result():
    X = BernoulliModel(p=0.5)
    Y = BernoulliModel(p=0.3)
    Z = BinaryFunctionModel(
        parent_signature=[BernoulliModel, BernoulliModel],
        f=lambda parent_values: np.prod(parent_values.astype(int), axis=1).astype(str),
    )

    causal_network = CausalNetwork({"X": X, "Y": Y, "Z": (Z, ["X", "Y"])})

    assert causal_network.draw_samples(10).shape == (10, 3)
    assert causal_network.draw_samples_with_noise(10).shape == (10, 6)
    assert causal_network.draw_noise_samples(10).shape == (10, 3)
    assert causal_network.get_node_value('Z', {'_X': "1", '_Y': "0"}) == "0"
