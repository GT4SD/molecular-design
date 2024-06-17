#!/usr/bin/env python3
"""MoLeR molecular generation."""
import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
from gt4sd.algorithms.conditional_generation.regression_transformer import (
    RegressionTransformer,
    RegressionTransformerMolecules,
)
from paccmann_generator.drug_evaluators import AromaticRing
from tqdm import tqdm

from helpers.molecules import decorate_smiles, parse_rt_samples, CANONICAL_DECORATIONS

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smi_path",
    "-s",
    type=str,
    help="Path to the SMILES data (.tsv). Has to contain SMILES in a `SMILES` column. "
    "Further columns include (predicted) pIC50 values based on kinase columns",
)
parser.add_argument(
    "--param_path",
    "-p",
    type=str,
    help="Path to parameter config file (json) for the generation.",
)
parser.add_argument("--output_path", "-o", type=str, help="Path to output directory")


def main(smi_path, param_path, output_path):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("rt_generator")
    logger.setLevel(level=logging.INFO)
    fh = logging.StreamHandler()
    fh_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(lineno)d:%(filename)s(%(process)d) - %(message)s"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    logger.info("===Starting script===")
    os.makedirs(output_path, exist_ok=True)

    # Extract params
    with open(param_path, "r") as f:
        params = json.load(f)

    version = params["rt_version"]
    max_to_mask = params["max_to_mask"]

    # Optional parameters
    batch_size = params.get("batch_size", 5)
    temperature = params.get("temperature", 1.25)
    tolerance = params.get("tolerance", 60)
    decorate = params.get("decorate", False)
    decorate_keep_prob = params.get("decorate_keep_prob", 0.7)

    params["smi_filepath"] = smi_path
    params["batch_size"] = batch_size
    params["decorate"] = decorate
    params["temperature"] = temperature
    params["tolerance"] = tolerance
    params["decorate_keep_prob"] = decorate_keep_prob

    filename = smi_path.split("/")[-1].split(".")[0]
    config_name = param_path.split("/")[-1].split(".")[0]
    output_folder = os.path.join(output_path, f"{version}_{config_name}_{filename}")
    os.makedirs(output_folder, exist_ok=True)
    config = RegressionTransformerMolecules(algorithm_version=version)
    model_path = config.ensure_artifacts()
    with open(os.path.join(model_path, "inference.json"), "r") as f:
        inference = json.load(f)
    properties = list(inference["property_ranges"].keys())

    logger.info("Reading data..")
    df = pd.read_csv(smi_path)

    # Set up oracles
    is_aromatic = AromaticRing()
    results = []
    with open(os.path.join(output_folder, "config.json"), "w") as f:
        json.dump(params, f, indent=4)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        to_mask = np.random.uniform(low=0.1, high=max_to_mask)
        # Try to push the property 10% upwards, but make sure value does not go beyond
        # 98% of maximum
        prop_goals = [
            round(
                max(float(row[k]) + 0.1 * (v[1] - v[0]), v[1] - (0.02 * (v[1] - v[0]))),
                3,
            )
            for k, v in inference["property_ranges"].items()
        ]
        sampling_wrapper = {
            "fraction_to_mask": to_mask,
            "property_goal": dict(zip(properties, prop_goals)),
        }
        config = RegressionTransformerMolecules(
            algorithm_version=version,
            search="sample",
            temperature=temperature,
            tolerance=tolerance,
            sampling_wrapper=sampling_wrapper,
        )
        model = RegressionTransformer(configuration=config, target=row.SMILES)
        try:
            samples = list(model.sample(batch_size))
        except Exception as e:
            logger.error(e)
            continue

        result = parse_rt_samples(samples)
        for entry in result:
            entry["seed"] = row.SMILES
            entry["to_mask"] = to_mask
            entry["decorate"] = False
        results.extend(result)

        if decorate:
            rt_language = config.generator.tokenizer.language
            if rt_language == "SELFIES":
                mask_structs = ["[Se]"]
            elif rt_language == "SMILES":
                mask_structs = CANONICAL_DECORATIONS
            elif decorate:
                raise ValueError(f"Unsupported language for decoration {rt_language}")
            # Manual masking. Convert to SELFIES and add mask tokens next to rings
            decorated = decorate_smiles(
                row.SMILES, keep_prob=decorate_keep_prob, rt_language=rt_language
            )
            sampling_wrapper.update(
                {
                    "substructures_to_mask": mask_structs,
                    "fraction_to_mask": 0.0,
                    "substructures_to_keep": [row.SMILES],
                }
            )
            config = RegressionTransformerMolecules(
                algorithm_version=version,
                search="sample",
                temperature=temperature,
                tolerance=tolerance,
                sampling_wrapper=sampling_wrapper,
            )
            model = RegressionTransformer(configuration=config, target=decorated)
            try:
                samples = list(model.sample(batch_size))
            except Exception as e:
                logger.error(e)
                continue
            result = parse_rt_samples(samples)
            for entry in result:
                entry["seed"] = row.SMILES
                entry["to_mask"] = to_mask
                entry["decorate"] = True
            results.extend(result)

    result_df = pd.DataFrame(results, index=range(len(results)))

    logger.info(f"{len(result_df)} molecules were generated.")
    result_df = result_df.drop_duplicates(subset="SMILES")
    logger.info(f"{len(result_df)} molecules were unique.")
    # Remove molecules without aromatic ring
    result_df["aromatic"] = result_df.SMILES.apply(lambda x: bool(is_aromatic(x)))
    result_df = result_df[result_df.aromatic].drop(["aromatic"], axis=1)
    logger.info(f"{len(result_df)} aromatic molecules.")
    result_df.to_csv(os.path.join(output_folder, "generated.csv"), index=False)
    logger.info("Done, shutting down...")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.smi_path, args.param_path, args.output_path)
