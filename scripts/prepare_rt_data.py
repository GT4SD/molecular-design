import pandas as pd
import argparse

from helpers.properties import qed_from_smiles

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smi_path", "-s", type=str, help="Path to generated SMILES csv file."
)
parser.add_argument(
    "--output_path", "-o", type=str, help="Path to the output csv file."
)


def calc_properties(output_path: str) -> None:
    df = pd.read_csv(args.smi_path)
    df["<qed>"] = df["SMILES"].apply(qed_from_smiles)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    calc_properties(args.output_path)
