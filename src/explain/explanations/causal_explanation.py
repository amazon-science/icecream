# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# Annotations and typing
from __future__ import annotations
from typing import Union, Any, Set, Optional, List

# Standard modules
import math
import itertools

# External modules
import pandas as pd
import numpy as np

# Internal modules
from explain.causal_network import CausalNetwork
from explain.causal_model import (
    FunctionCausalModel,
    FiniteNoiseCausalModel,
    DeterministicCausalModel,
)
from explain.utils import (
    list_coalitions,
    noisify,
)

from explain.probability import (
    recover_noise_distribution,
    noise_probability_table,
    expected_value,
)


ExplainableCausalModel = Union[FunctionCausalModel, FiniteNoiseCausalModel]


def exact_explanation_score(
    causal_network: CausalNetwork[ExplainableCausalModel],
    observation: pd.Series,
    target: Any,
    coalition: Set[Any],
) -> pd.DataFrame:
    """Calculates the exact explanation score distribution of a given observation,
    coalition and target node.

    :param causal_network: The underlying causal network.
    :param observation: The observation for which the explanation score is calculated.
    :param target: The target node whose distribution is considered.
    :param coalition: The coalition of nodes for which the explanation score is
    calculated.
    :return: A dataframe containing the probability distribution of the explanation
    score for the coalition.
    """

    # The empty coalition always has explanation score 0
    if not coalition:
        return pd.DataFrame({"__p": [1.0], "__s": [0.0]})

    # Find all noise combinations which are compatible with the observation, and their
    # respective probabilities.
    noise_distribution = recover_noise_distribution(
        causal_network, observation.to_dict()
    )

    if noise_distribution.empty:
        raise ValueError(
            f"The observation {observation} does not correspond to any noise values!"
        )
    else:
        probability_table = noise_probability_table(
            causal_network, include_node_values=True
        )

        # The base probability is P[Y = y] (which is the same for all noise
        # combinations)
        base_probability = probability_table["__p"][
            probability_table[target] == observation[target]
        ].sum()
        if base_probability in {0, 1}:
            raise ValueError(
                f"The explanation score is not defined if P[{target}"
                f" = {observation[target]}] = {base_probability}!"
            )

        coalition_noise_distribution = (
            noise_distribution[noisify(coalition) + ["__p"]]
            .groupby(noisify(coalition))
            .sum()
            .reset_index()
        )
        coalition_noise_distribution["__s"] = None

        for index, coalition_noise in coalition_noise_distribution.iterrows():
            interventional_table = probability_table[
                (
                    probability_table[noisify(coalition)]
                    == coalition_noise[noisify(coalition)]
                ).values.all(axis=1)
            ]

            # The interventional probability is P[Y = y | do(X_C = x_C)]
            interventional_probability = (
                interventional_table["__p"][
                    interventional_table[target] == observation[target]
                ].sum()
                / interventional_table["__p"].sum()
            )

            # All coalition noise combinations have positive probability, so the
            # expected value is -inf if at least
            # one of them makes it impossible to obtain the target value
            if interventional_probability == 0:
                raise ValueError(
                    f"The intervention on {coalition} makes observing "
                    f"{observation} impossible!"
                )

            coalition_noise_distribution.at[index, "__s"] = 1.0 - math.log(
                interventional_probability
            ) / math.log(base_probability)

        return coalition_noise_distribution[["__p", "__s"]]


def expected_exact_explanation_scores(
    causal_network: CausalNetwork[ExplainableCausalModel],
    observation: pd.Series,
    target: Any,
    coalitions: Optional[List[Set[Any]]] = None,
) -> pd.DataFrame:
    """Calculates the expected values of the exact explanation score distributions of a
    given observation and target
    for all (or a given list of) coalitions.

    The explanation score is 1 - log(P[Y = y]) / log(P[Y = y | do(X_C = x_C)]), as
    defined in Oesterle et al. (2023): Beyond Single Feature Importance: Finding
    Optimal Coalitions for Explainable AI (https://arxiv.org/pdf/XXX.pdf)

    :param causal_network: The underlying causal network.
    :param observation: The observation for which the explanation score is calculated.
    :param target: The target node whose distribution is considered.
    :param coalitions: A list of coalitions for which the explanation score is
    calculated. If None, all coalitions of
    the causal network are used.
    :return: A dataframe containing the expected explanation score for all coalitions.
    """

    # Find all noise combinations which are compatible with the observation, and their
    # respective probabilities.
    noise_distribution = recover_noise_distribution(
        causal_network, observation.to_dict()
    )

    if noise_distribution.empty:
        raise ValueError(
            f"The observation {observation} does not correspond to any noise values!"
        )

    probability_table = noise_probability_table(
        causal_network, include_node_values=True
    )

    # The base probability is P[Y = y] (which is the same for all coalitions)
    base_probability = probability_table["__p"][
        probability_table[target] == observation[target]
    ].sum()
    if base_probability in {0, 1}:
        raise ValueError(
            f"The explanation score is not defined if "
            f"P[{target} = {observation[target]}] = {base_probability}!"
        )

    if coalitions is None:
        coalitions = list_coalitions(causal_network)

    explanation_scores = pd.DataFrame(
        index=range(len(coalitions)), columns=["coalition", "__s"]
    )
    for coalition_index, coalition in enumerate(coalitions):
        if not coalition:
            # The empty coalition always has explanation score 0
            explanation_scores.iloc[coalition_index] = {
                "coalition": coalition,
                "__s": 0.0,
            }
        else:
            coalition_noise_distribution = (
                noise_distribution[noisify(coalition) + ["__p"]]
                .groupby(noisify(coalition))
                .sum()
                .reset_index()
            )
            coalition_noise_distribution["__s"] = None

            for index, coalition_noise in coalition_noise_distribution.iterrows():
                interventional_table = probability_table[
                    (
                        probability_table[noisify(coalition)]
                        == coalition_noise[noisify(coalition)]
                    ).values.all(axis=1)
                ]

                # The interventional probability is P[Y = y | do(X_C = x_C)]
                interventional_probability = (
                    interventional_table["__p"][
                        interventional_table[target] == observation[target]
                    ].sum()
                    / interventional_table["__p"].sum()
                )
                if interventional_probability == 0:
                    coalition_noise_distribution = pd.DataFrame(
                        {"__p": [1.0], "__s": [-np.inf]}
                    )
                    break

                coalition_noise_distribution.at[index, "__s"] = 1.0 - math.log(
                    interventional_probability
                ) / math.log(base_probability)

            explanation_scores.iloc[coalition_index] = {
                "coalition": coalition,
                "__s": expected_value(coalition_noise_distribution),
            }

    return explanation_scores


def find_minimum_size_coalitions(
    causal_network: CausalNetwork[ExplainableCausalModel],
    observation: pd.Series,
    target: Any,
    threshold: float = 0.998,
    exclude_deterministic_nodes: bool = True,
    nodes_to_exclude: Set[Any] = None,
) -> List[Set[Any]]:
    """Finds all minimum-size coalitions whose exact explanation score is above a given
    threshold.

    The explanation score is 1 - log(P[Y = y]) / log(P[Y = y | do(X_C = x_C)]), as
    defined in Oesterle et al. (2023): Beyond Single Feature Importance: Finding
    Optimal Coalitions for Explainable AI (https://arxiv.org/pdf/XXX.pdf)

    :param causal_network: The underlying causal network.
    :param observation: The observation for which the explanation score is calculated.
    :param target: The target node whose distribution is considered.
    :param threshold: The lower bound for the explanation score.
    :param exclude_deterministic_nodes: If true, only consider non-deterministic (i.e.,
    noisy) nodes for coalitions.
    :param nodes_to_exclude: Set of nodes which should not be considered for coalitions.
    :return: A list of all minimal coalitions with explanation score greater or equal
    to the threshold.
    """

    # Prepare coalitions to iterate over
    if nodes_to_exclude is None:
        nodes_to_exclude = set()

    if exclude_deterministic_nodes:
        nodes_to_exclude = nodes_to_exclude.union(
            set(
                node
                for node, causal_model in causal_network
                if isinstance(causal_model, DeterministicCausalModel)
            )
        )

    coalition_nodes = set(causal_network.nodes) - nodes_to_exclude

    # Find all noise combinations which are compatible with the observation, and their
    # respective probabilities.
    noise_distribution = recover_noise_distribution(
        causal_network, observation.to_dict()
    )

    if noise_distribution.empty:
        raise ValueError(
            f"The observation {observation} does not correspond to any noise values!"
        )

    probability_table = noise_probability_table(
        causal_network, include_node_values=True
    )

    # The base probability is P[Y = y] (which is the same for all coalitions)
    base_probability = probability_table["__p"][
        probability_table[target] == observation[target]
    ].sum()
    if base_probability == 0:
        raise ValueError(f"The target value {observation[target]} has probability 0!")

    # Iterate over coalitions in ascending order of size
    for coalition_size in range(1, len(coalition_nodes) + 1):
        minimal_coalitions = []
        for coalition in map(
            set, itertools.combinations(coalition_nodes, coalition_size)
        ):
            coalition_noise_distribution = (
                noise_distribution[noisify(coalition) + ["__p"]]
                .groupby(noisify(coalition))
                .sum()
                .reset_index()
            )
            coalition_noise_distribution["__s"] = None

            for index, coalition_noise in coalition_noise_distribution.iterrows():
                interventional_table = probability_table[
                    (
                        probability_table[noisify(coalition)]
                        == coalition_noise[noisify(coalition)]
                    ).values.all(axis=1)
                ]

                # The interventional probability is P[Y = y | do(X_C = x_C)]
                interventional_probability = (
                    interventional_table["__p"][
                        interventional_table[target] == observation[target]
                    ].sum()
                    / interventional_table["__p"].sum()
                )
                if interventional_probability == 0:
                    coalition_noise_distribution = pd.DataFrame(
                        {"__p": [1.0], "__s": [-np.inf]}
                    )
                    break

                coalition_noise_distribution.at[index, "__s"] = 1.0 - math.log(
                    interventional_probability
                ) / math.log(base_probability)

            if expected_value(coalition_noise_distribution) > threshold:
                minimal_coalitions.append(coalition)

        if minimal_coalitions:
            return minimal_coalitions

    raise ValueError(
        f"No minimal coalition with explanation score >{threshold} was found among "
        f"nodes {coalition_nodes} for the observation {observation}!"
    )
