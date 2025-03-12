# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# Annotations and typing
from __future__ import annotations
from typing import Any, Set, Optional

# Standard modules
import itertools
import math
import func_timeout

# External modules
import pandas as pd


def estimate_explanation_score(
    samples: pd.DataFrame,
    observation: pd.Series,
    target: Any,
    coalition: Set[Any],
    verbose: bool = False,
) -> float:
    """Estimates the explanation score of a given coalition from the provided data.

    The explanation score is 1 - log(P[Y = y]) / log(P[Y = y | do(X_C = x_C)]), as
    defined in Oesterle et al. (2023): Beyond Single Feature Importance: Finding
    Optimal Coalitions for Explainable AI (https://arxiv.org/pdf/XXX.pdf)

    :param samples: Samples from the joint distribution of X and Y.
    :param observation: Observation to be explained by the coalition.
    :param target: Identifier of the target.
    :param coalition: Set of features for which the explanation score is estimated.
    :param verbose: If true, print status information.
    :return: The explanation score.
    """
    base_samples = samples[target] == observation[target]
    base_probability = base_samples.mean()

    if verbose:
        print(f"{base_probability=} ({base_samples.sum()}/{len(base_samples)})")

    if base_probability in {0, 1}:
        raise ValueError(
            f"The explanation score is not defined if P[{target}"
            f" = {observation[target]}] = {base_probability}!"
        )

    conditional_samples = (
        samples[target][
            (samples[list(coalition)] == observation[list(coalition)]).values.all(
                axis=1
            )
        ]
        == observation[target]
    )
    conditional_probability = conditional_samples.mean()

    if verbose:
        print(
            f"{conditional_probability=} (={conditional_samples.sum()}"
            f"/{len(conditional_samples)})"
        )

    if conditional_probability == 0:
        raise ValueError(
            f"Conditioning on {coalition} makes observing {observation} impossible!"
        )
    else:
        return 1.0 - math.log(conditional_probability) / math.log(base_probability)


def _find_minimum_size_coalitions(
    samples: pd.DataFrame,
    observation: pd.Series,
    target: Any,
    threshold: float = 0.998,
    features_to_exclude: Set[Any] = None,
    minimum_coalition_size: int = 1,
    maximum_coalition_size: Optional[int] = None,
    return_at_first_result: bool = False,
    verbose: bool = False,
) -> Set[frozenset[Any]]:
    """Lists all coalitions with minimal size whose explanation score exceeds the given
    threshold. In other words, if there is no 2-feature coalition with explanation
    score >= threshold, but a 3-feature coalition, the function returns _all_ such
    coalitions with 3 features.

    The explanation score is 1 - log(P[Y = y]) / log(P[Y = y | do(X_C = x_C)]), as
    defined in Oesterle et al. (2023): Beyond Single Feature Importance: Finding
    Optimal Coalitions for Explainable AI (https://arxiv.org/pdf/XXX.pdf)

    :param samples: Samples from the joint distribution of X and Y.
    :param observation: Observation to be explained by the coalitions.
    :param target: Identifier of the target.
    :param threshold: Lower bound for the explanation score of a coalition.
    :param features_to_exclude: Set of features that are ignored for the coalitions.
    :param minimum_coalition_size: Lower bound for the size of a coalition.
    :param maximum_coalition_size: Upper bound (included) for the size of a coalition.
    :param return_at_first_result: If true, return as soon as a coalition with
    explanation score >= threshold is found.
    If False, search all coalitions of minimal size before returning.
    :param verbose: If true, print status information.
    :return: A set of all coalitions with minimal size whose explanation score exceeds
    the given threshold.
    """
    if features_to_exclude is None:
        features_to_exclude = {target}

    coalition_features = set(samples.columns) - features_to_exclude

    if maximum_coalition_size is None:
        maximum_coalition_size = len(coalition_features)

    base_samples = samples[target] == observation[target]
    base_probability = base_samples.mean()

    if verbose:
        print(f"{base_probability=} ({base_samples.sum()}/{len(base_samples)})")

    if base_probability in {0, 1}:
        raise ValueError(
            f"The explanation score is not defined if P[{target}"
            f" = {observation[target]}] = {base_probability}!"
        )

    for coalition_size in range(
        minimum_coalition_size, min(len(coalition_features), maximum_coalition_size + 1)
    ):
        minimal_coalitions = set()

        if verbose:
            print(f"Searching for coalitions of size {coalition_size}...")

        for coalition in map(
            set, itertools.combinations(coalition_features, coalition_size)
        ):
            conditional_samples = (
                samples[target][
                    (
                        samples[list(coalition)] == observation[list(coalition)]
                    ).values.all(axis=1)
                ]
                == observation[target]
            )
            conditional_probability = conditional_samples.mean()

            if (conditional_probability > 0) and (
                1.0 - math.log(conditional_probability) / math.log(base_probability)
                >= threshold
            ):
                if verbose:
                    print(
                        f"{conditional_probability=} ({conditional_samples.sum()}"
                        f"/{len(conditional_samples)})"
                    )

                minimal_coalitions.add(frozenset(coalition))

                if return_at_first_result:
                    break

        if minimal_coalitions:
            if verbose:
                print()
            return minimal_coalitions
        else:
            if verbose:
                print("  None found!")

    return set()


def find_minimum_size_coalitions(
    samples: pd.DataFrame,
    observation: pd.Series,
    target: Any,
    threshold: float = 0.998,
    features_to_exclude: Set[Any] = None,
    minimum_coalition_size: int = 1,
    maximum_coalition_size: Optional[int] = None,
    timeout: Optional[float] = None,
    return_at_first_result: bool = False,
    verbose: bool = False,
) -> Set[frozenset[Any]]:
    """Lists all coalitions with minimal size whose explanation score exceeds the given
    threshold. In other words, if there is no 2-feature coalition with explanation
    score >= threshold, but a 3-feature coalition, the function returns _all_ such
    coalitions with 3 features.

    The explanation score is 1 - log(P[Y = y]) / log(P[Y = y | do(X_C = x_C)]), as
    defined in Oesterle et al. (2023): Beyond Single Feature Importance: Finding
    Optimal Coalitions for Explainable AI (https://arxiv.org/pdf/XXX.pdf)

    :param samples: Samples from the joint distribution of X and Y.
    :param observation: Observation to be explained by the coalitions.
    :param target: Identifier of the target.
    :param threshold: Lower bound for the explanation score of a coalition.
    :param features_to_exclude: Set of features that are ignored for the coalitions.
    :param minimum_coalition_size: Lower bound for the size of a coalition.
    :param maximum_coalition_size: Upper bound (included) for the size of a coalition.
    :param timeout: Number of seconds before the function returns. If the timeout hits,
    an exception is raised.
    :param return_at_first_result: If true, return as soon as a coalition with
    explanation score >= threshold is found.
    If False, search all coalitions of minimal size before returning.
    :param verbose: If true, print status information.
    :return: A set of all coalitions with minimal size whose explanation score exceeds
    the given threshold.
    """
    if timeout is not None:
        try:
            return func_timeout.func_timeout(
                timeout,
                _find_minimum_size_coalitions,
                [samples, observation],
                {
                    "target": target,
                    "threshold": threshold,
                    "features_to_exclude": features_to_exclude,
                    "minimum_coalition_size": minimum_coalition_size,
                    "maximum_coalition_size": maximum_coalition_size,
                    "return_at_first_result": return_at_first_result,
                    "verbose": verbose,
                },
            )
        except func_timeout.FunctionTimedOut:
            return set()
    else:
        return _find_minimum_size_coalitions(
            samples,
            observation,
            target,
            threshold,
            features_to_exclude,
            minimum_coalition_size,
            maximum_coalition_size,
            return_at_first_result=return_at_first_result,
            verbose=verbose,
        )
