# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
import pandas as pd
import numpy as np

from explain.explanations.data_explanation import (
    estimate_explanation_score,
    find_minimum_size_coalitions,
)


def test_given_data_then_explanation_score_returns_expected_result():
    samples = pd.DataFrame(
        np.random.randint(0, 2, size=(100, 3)), columns=["X", "Y", "Z"]
    )

    estimate_explanation_score(samples, samples.iloc[0], "Z", {"X", "Y"})

    with pytest.raises(ValueError):
        estimate_explanation_score(
            samples, pd.Series({"X": 0, "Y": 1, "Z": 2}), "Z", {"X", "Y"}
        )

    assert (
        estimate_explanation_score(samples, samples.iloc[0], "Z", {"X", "Y", "Z"}) == 1
    )
    assert estimate_explanation_score(samples, samples.iloc[0], "Z", {}) == 0

    estimate_explanation_score(
        samples.astype(str), samples.iloc[0].astype(str), "Z", {"X", "Y"}
    )


def test_given_data_then_find_minimum_size_coalitions_returns_expected_result():
    samples = pd.DataFrame(
        np.random.randint(0, 2, size=(100, 3)), columns=["X", "Y", "Z"]
    )

    find_minimum_size_coalitions(samples, samples.iloc[0], "Z")
