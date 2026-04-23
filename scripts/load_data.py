from typing import Optional
import requests
import os
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from helpers import utils
from time import sleep


parser = argparse.ArgumentParser()
parser.add_argument(
    "--uniprot", "-u", type=str, help="UniProt code for protein to fetch."
)
parser.add_argument(
    "--output_dir",
    "-o",
    default="data/",
    type=str,
    help="The directory where datasets are saved.",
)
parser.add_argument(
    "--affinity_cutoff", type=int, default=10_000, help="Affinity cutoff point."
)
parser.add_argument(
    "--affinity_type",
    type=str,
    default="IC50",
    choices=["IC50", "Kd"],
    help="IC50 or Kd",
)
parser.add_argument(
    "--train_size",
    type=float,
    default=0.8,
    help="Proportion of dataset included in the train split.",
)
parser.add_argument(
    "--binary_labels",
    action="store_true",
    help="Enable binary classification. If not specified, the default mode is regression.",
)
parser.add_argument(
    "--max_retries",
    type=int,
    default=10,
    help="Maximal number of retries to fetch data from fickle BindingDB API",
)


def fetch(
    uniprot: str,
    affinity_cutoff: int,
    affinity_type: str,
) -> Optional[pd.DataFrame]:
    url = f"https://www.bindingdb.org/rest/getLigandsByUniprots?uniprot={uniprot}&cutoff={affinity_cutoff}&response=application/json"
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    data = response.json()
    if 'getLindsByUniprotsResponse' not in data:
        return
    affinities = data["getLindsByUniprotsResponse"]["affinities"]
    df = pd.DataFrame(affinities)
    df = df[df["affinity_type"] == affinity_type].copy()
    df = df[["smile", "monomerid", "affinity"]].dropna()
    if df.empty:
        return df
    df["affinity"] = df["affinity"].apply(utils.str2float)
    df["affinity"] = df["affinity"].apply(utils.nanomolar_to_pic50)

    # BindingDB can return multiple assay records and non-unique monomer IDs for
    # the same molecule. ToxSmi only needs a unique molecule ID, so aggregate
    # repeated SMILES and assign stable IDs for this downloaded dataset.
    df = df.groupby("smile", as_index=False)["affinity"].median()
    df = df.sort_values("smile").reset_index(drop=True)
    df["monomerid"] = [f"mol_{i}" for i in range(len(df))]
    df = df[["smile", "monomerid", "affinity"]]
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    args = parser.parse_args()

    dataset = None
    last_error = None
    for attempt in range(args.max_retries):
        try:
            dataset = fetch(args.uniprot, args.affinity_cutoff, args.affinity_type)
        except requests.RequestException as error:
            last_error = error
            dataset = None
            print(f"Attempt {attempt + 1}/{args.max_retries} failed: {error}")

        if dataset is not None and not dataset.empty:
            break
        sleep(5)

    if dataset is None or dataset.empty:
        message = (
            f"BindingDB returned no usable {args.affinity_type} data for "
            f"{args.uniprot} after {args.max_retries} attempts."
        )
        if last_error is not None:
            message += f" Last error: {last_error}"
        raise SystemExit(message)
    else:
        # three files. mols.smi list of all the smiles. Then we have train.csv and val.csv
        os.makedirs(args.output_dir, exist_ok=True)
        mol_path = os.path.join(args.output_dir, "mols.smi")
        train_path = os.path.join(args.output_dir, "train.csv")
        val_path = os.path.join(args.output_dir, "valid.csv")
        # Save smiles and id without header. Note that this dataset uses tab delimiter.
        dataset[["smile", "monomerid"]].to_csv(
            mol_path, index=False, header=False, sep="\t"
        )
        # Training dataset have columns Label,sampling_frequency,mol_id
        dataset = dataset.rename(columns={"affinity": "Label", "monomerid": "mol_id"})
        dataset["sampling_frequency"] = "high"
        dataset = dataset[["Label", "sampling_frequency", "mol_id"]]

        if args.binary_labels:
            dataset["Label"] = dataset["Label"].apply(lambda x: 1 if x > 6 else 0)
        train, validation = train_test_split(
            dataset, train_size=args.train_size, random_state=1911
        )
        train.to_csv(train_path, index=False, header=True)
        validation.to_csv(val_path, index=False, header=True)
