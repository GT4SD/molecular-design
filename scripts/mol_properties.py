import pandas as pd
import argparse

from helpers.properties import compute_rdkit_properties

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smi_path",
    "-s",
    type=str,
    help="Path to the csv file containing a 'SMILES' column that properties should be calculated for.",
)
parser.add_argument(
    "--output_path", "-o", type=str, help="Path to the output csv file."
)


def calc_properties(output_path: str) -> None:
    mols = pd.read_csv(args.smi_path)
    smiles = mols[["SMILES"]].copy()
    if "Prediction" in mols.columns:
        smiles["IC50"] = mols["Prediction"]
    elif "pred_0" in mols.columns:
        smiles["IC50"] = mols["pred_0"]

    props = mols["SMILES"].apply(
        lambda smiles_value: pd.Series(
            compute_rdkit_properties(smiles_value), dtype="float64"
        )
    )
    smiles = pd.concat([smiles, props], axis=1)

    smiles.to_csv(output_path, index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    calc_properties(args.output_path)
