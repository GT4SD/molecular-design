#!/usr/bin/env python3
"""MoLeR molecular generation."""
import argparse
import json
import logging
import os
from paccmann_predictor.utils.utils import get_device
from toxsmi.utils.performance import PerformanceLogger
from sklearn.metrics import fbeta_score, classification_report
from scipy.stats import hmean
import numpy as np
import pandas as pd
from helpers.molecules import extract_motifs_from_smiles_list
from paccmann_generator.drug_evaluators import AromaticRing
from helpers.toxsmi import restore_toxsmi, get_loader
from toxsmi.utils.hyperparams import ACTIVATION_FN_FACTORY
from pytoda.smiles.transforms import Canonicalization
from tqdm import tqdm
from helpers.utils import get_tani

from gt4sd.algorithms.generation.moler import MoLeR, MoLeRDefaultGenerator

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smi_path",
    "-s",
    type=str,
    help="Path to the SMILES data (.tsv). Has to contain SMILES in a `SMILES` column. "
    "Can optionally contain labels in a `Labels` column in which case the average label "
    "of motif is used instead of the relative frequency.",
)
parser.add_argument(
    "--param_path", "-p", type=str, help="Path to parameter config file (json)."
)
parser.add_argument("--output_path", "-o", type=str, help="Path to output directory")
parser.add_argument(
    "--predictor_path",
    "-pp",
    nargs="?",
    default=None,
    help="Path to a checkpoint of a pretrained affinity predictor. If not provided, "
    "no filtering is done.",
)


def main(smi_path, param_path, output_path, predictor_path):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("moler_generator")
    logger.setLevel(level=logging.INFO)
    fh = logging.StreamHandler()
    fh_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(lineno)d:%(filename)s(%(process)d) - %(message)s"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    logger.info("===Starting script===")
    device = get_device()

    os.makedirs(output_path, exist_ok=True)
    tmp_path = os.path.join(output_path, "tmp")
    os.makedirs(tmp_path, exist_ok=True)

    # Extract params
    with open(param_path, "r") as f:
        params = json.load(f)

    num_mols = params["mols_per_iteration"]
    max_sigma = params["max_sigma"]
    motif_extractor = params["motif_extractor"]
    iterations = params["iterations"]

    # Optional parameters
    batch_size = params.get("batch_size", 128)
    sample_size = params.get("sample_size", params.get("sample_size", 128) / 2)
    workers = params.get("workers", 1)
    beam_size = params.get("beam_size", 1)
    seed = params.get("seed", 42)
    swap_order = params.get("swap_order", False)
    theta = params.get("theta", 3.5)
    pbs = params.get("predictor_batch_size", 8)

    params["smi_filepath"] = smi_path
    params["genetic_optimization"] = predictor_path is not None

    logger.info("Reading data..")
    df = pd.read_csv(smi_path, sep="\t", header=None)
    smiles = list(df[0])

    # Model should be fed with positive samples only
    labels = None

    # Set up oracles
    is_aromatic = AromaticRing()

    if predictor_path:
        predictor, smiles_language = restore_toxsmi(predictor_path)
        predictor.final_dense.act_fn = ACTIVATION_FN_FACTORY["sigmoid"]
    else:
        predictor = None

    # initialize the input for motif extraction to the input smiles
    all_molecules = set()
    in_molecules = smiles
    result_df = []
    can = Canonicalization()

    for i in range(1, iterations + 1):
        logger.info(f"Starting iteration {i} with {len(in_molecules)} molecules")

        # For later generations, we dont have label information anymore
        if i > 0:
            labels = None

        # Extract the motifs
        all_scaffolds, scaffold_scores = extract_motifs_from_smiles_list(
            in_molecules,
            extractor_string="scaffold",
            labels=labels,
            filter_atoms=["Na"],
        )
        all_motifs, motif_scores = extract_motifs_from_smiles_list(
            in_molecules,
            extractor_string=motif_extractor,
            labels=labels,
            filter_atoms=["Na"],
        )
        if params.get("add_seeds_to_scaffolds", False):
            all_scaffolds.extend(in_molecules)
            scaffold_scores = np.concatenate([scaffold_scores, [2] * len(in_molecules)])

        if params.get("add_seeds_to_motifs", False):
            all_motifs.extend(in_molecules)
            motif_scores = np.concatenate([motif_scores, [2] * len(in_molecules)])

        for s in all_scaffolds:
            if "." in s:
                raise ValueError(s)
        for m in all_motifs:
            if "." in m:
                raise ValueError(s)

        # Normalize the scores to one
        scaffold_scores /= sum(scaffold_scores)
        motif_scores /= sum(motif_scores)

        molecules = set()
        logger.info(
            f"Starting to generate now, {len(all_scaffolds)} scaffolds and {len(all_motifs)} motifs"
        )

        runs = 0
        while len(molecules) < num_mols:
            # Start a call
            sigma = np.random.uniform(low=0.1, high=max_sigma)
            scaffolds = np.random.choice(
                all_scaffolds, batch_size, replace=True, p=scaffold_scores
            )
            motifs = np.random.choice(
                all_motifs, batch_size, replace=True, p=motif_scores
            )

            scaffolds = ".".join(scaffolds)
            motifs = ".".join(motifs)

            config = MoLeRDefaultGenerator(
                scaffolds=scaffolds,
                beam_size=beam_size,
                num_samples=batch_size,
                seed=seed,
                num_workers=workers,
                seed_smiles=motifs,
                sigma=sigma,
            )
            model = MoLeR(configuration=config)
            try:
                samples = set(model.sample(sample_size))
                molecules = molecules.union(samples)
            except RuntimeError as e:
                logger.error(f"Worker process died: {e}")
                continue
            if swap_order:
                config = MoLeRDefaultGenerator(
                    scaffolds=motifs,
                    beam_size=beam_size,
                    num_samples=batch_size,
                    seed=seed,
                    num_workers=workers,
                    seed_smiles=scaffolds,
                    sigma=sigma,
                )
                model = MoLeR(configuration=config)
                try:
                    samples = set(model.sample(sample_size))
                    molecules = molecules.union(samples)
                except RuntimeError as e:
                    logger.error(f"Worker process died: {e}")
                    continue

            seed += 1
            runs += 1
            logger.info(
                f"====={runs} runs completed. Currently {len(molecules)} molecules====="
            )
        logger.info(f"Completed sampling iteration {i}. Got {len(molecules)} molecules")

        # Remove molecules without aromatic ring
        molecules = list(filter(is_aromatic, molecules))
        molecules = list(map(can, molecules))
        logger.info(f"{len(molecules)} aromatic molecules.")

        all_molecules = all_molecules.union(molecules)

        # Predict molecules with a model of a predefined path
        if predictor is None and not params.get("design_linker", False):
            result_df.append(pd.DataFrame({"SMILES": molecules, "Iteration": i}))
            continue

        # Predict and filter molecules
        # Create dataset (molecule file)
        generated_path = os.path.join(tmp_path, f"generated_iteration_{i}.smi")
        mol_ids = [f"Iter_{i}_{j}" for j in range(len(molecules))]
        generated = pd.DataFrame({"SMILES": molecules, "ID": mol_ids})
        generated.to_csv(generated_path, header=False, sep="\t", index=False)

        if params.get("design_linker", False):
            label_path = os.path.join(tmp_path, f"iteration_{i}_linker.csv")
            # Optimize harmonic mean of similarity to seeds
            hmeans = []
            for gen_smi in molecules:
                hmeans.append(
                    hmean([get_tani(gen_smi, seed_smi) for seed_smi in in_molecules])
                )
            # Sort by hmean and take everything above 0.3 (top 10% if all are below 0.3)
            best_molecules = np.array(molecules)[
                np.array(hmeans) > params.get("hmean_theta", 0.3)
            ]
            if len(best_molecules) < 5:
                logger.warning(
                    f"Only {len(best_molecules)} molecules passed the hmean filter."
                )
                # Take the 10% of molecules with the highest hmean
                best_molecules = np.array(list(molecules))[
                    np.argsort(hmeans)[-int(len(molecules) * 0.1) :]
                ].tolist()

            # Overwite mol_ids so that predictor only predicts the filtered molecules
            best_mol_ids = [
                m for i, m in enumerate(mol_ids) if molecules[i] in best_molecules
            ]
            hmeans = [h for i, h in enumerate(hmeans) if molecules[i] in best_molecules]
            post_df = pd.DataFrame(
                {
                    "SMILES": best_molecules,
                    "mol_id": best_mol_ids,
                    "HTani": hmeans,
                    "Iteration": i,
                }
            )
            post_df.to_csv(
                os.path.join(tmp_path, f"iteration_{i}_linker.csv"), index=True
            )
            logger.info(
                f"From {len(all_molecules)} molecules, after filtering based on hmean to seeds "
                f"{len(post_df)} entries remain."
            )
            post_df["Label"] = post_df.HTani
            result_df.append(post_df)

        if predictor is not None:
            # Create label file
            label_path = os.path.join(tmp_path, f"iteration_{i}_affinity.csv")
            # Avoid batches with one sample
            print(f"Predicting now {len(mol_ids)} molecules with {pbs} batch size")
            if len(mol_ids) % pbs == 1:
                mol_ids.append(mol_ids[-1])
            label_df = pd.DataFrame({"Label": -1, "mol_id": mol_ids})
            label_df.to_csv(label_path, index=False)
            dataset, loader = get_loader(
                label_path=label_path,
                smiles_path=generated_path,
                smiles_language=smiles_language,
                batch_size=pbs,
            )

            # Run predictions
            logger.info("Setup complete, running predictions")
            predictions = []
            for ind, (smiles, y) in tqdm(enumerate(loader), total=len(loader)):
                # This verifies that no augmentation occurs, shuffle is False and that the
                # order of the dataloder is identical to the dataset
                assert all(smiles[0, :] == dataset[ind * pbs][0])
                assert all(y[0, :] == dataset[ind * pbs][1])
                y_hat, pred_dict = predictor(smiles.to(device))
                predictions.extend(y_hat.cpu().detach().squeeze().tolist())

            # Filter out molecules with predictions below threshold
            label_df["Label"] = predictions
            label_df["SMILES"] = molecules
            # Overwrite with predictions
            label_df.to_csv(label_path, index=True)
            id_to_smi = dict(zip(label_df.mol_id, label_df.SMILES))
            label_df.index = list(range(len(label_df)))
            label_df["SMILES"] = [id_to_smi[x] for x in label_df.mol_id]
            post_df = label_df[label_df["Label"] > theta]
            post_df["Iteration"] = i
            num_positives = (label_df["Label"] > theta).sum().sum()
            post_df.to_csv(label_path.replace(".csv", "_effective.csv"), index=True)
            logger.info(
                f"From {len(label_df)} entries of {len(mol_ids)} molecules, after filtering "
                f"with Label={theta}, {num_positives} entries of {len(post_df)} remain."
            )
            result_df.append(post_df)

        # Assign variables for next round of genetic algorithm
        in_molecules = list(set(in_molecules).union(set(post_df.SMILES)))
        labels = None
        logger.info(f"Completed generation and filtering round {i}")

    logger.info("Completed generation")
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(params, f, indent=4)

    result_df = pd.concat(result_df, axis=0).drop_duplicates(subset="SMILES")

    result_df = result_df.sort_values(by="Label", ascending=False)
    result_df.to_csv(os.path.join(output_path, "generated.csv"), index=False)

    means = result_df.groupby("Iteration")["Label"].describe()["mean"]
    logger.info(f"Average Label per Iteration = {means}")

    logger.info("Done, shutting down...")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.smi_path, args.param_path, args.output_path, args.predictor_path)
