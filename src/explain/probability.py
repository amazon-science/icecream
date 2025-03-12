# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# Annotations and typing
from __future__ import annotations
from typing import Any, Dict

# Standard modules
import itertools

# External modules
import pandas as pd

# Internal modules
from explain.causal_network import CausalNetwork
from explain.causal_model import (
    NoiseCausalModel,
    NoiseFunctionCausalModel,
    NondeterministicCausalModel,
    FunctionCausalModel,
)
from explain.utils import dataframe_to_array, noisify


def noise_probability_table(
    causal_network: CausalNetwork, include_node_values: bool = False
) -> pd.DataFrame:
    table = pd.DataFrame(
        list(
            itertools.product(
                *(
                    causal_model.noise_support()
                    if isinstance(causal_model, NondeterministicCausalModel)
                    else {0}
                    for _, causal_model in causal_network
                )
            )
        ),
        columns=noisify(causal_network.nodes),
    )
    table["__p"] = pd.concat(
        [
            table[f"_{node}"].apply(causal_model.noise_probability)
            for node, causal_model in causal_network
            if isinstance(causal_model, NondeterministicCausalModel)
        ]
        + [pd.Series(1, index=table.index)],
        axis=1,
    ).prod(axis=1)

    if include_node_values:
        for node, causal_model in causal_network:
            if isinstance(causal_model, NoiseCausalModel):
                table[node] = causal_model.evaluate(
                    dataframe_to_array(table, f"_{node}")
                )
            elif isinstance(causal_model, FunctionCausalModel):
                table[node] = causal_model.draw_samples(
                    dataframe_to_array(table, causal_network.parents(node))
                )
            elif isinstance(causal_model, NoiseFunctionCausalModel):
                table[node] = causal_model.evaluate(
                    dataframe_to_array(table, causal_network.parents(node)),
                    dataframe_to_array(table, f"_{node}"),
                )
            else:
                raise ValueError(f"{causal_model} has an invalid type!")

    return table


def recover_noise_distribution(
    causal_network: CausalNetwork, sample: Dict[Any, float]
) -> pd.DataFrame:
    table = noise_probability_table(causal_network, include_node_values=True)
    conditionals = pd.Series(sample)
    table = table.loc[
        (table[conditionals.index] == conditionals).all(axis=1)
    ].reset_index(drop=True)
    table["__p"] /= table["__p"].sum()
    table = table.loc[table["__p"] > 0.0].reset_index(drop=True)

    return table


def expected_value(table: pd.DataFrame) -> float:
    return (table["__p"] * table["__s"]).sum()
