import pandas as pd
import argparse
from gt4sd.properties import PropertyPredictorRegistry

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smi_path",
    "-s",
    type=str,
    help="Path to the csv file containing a 'SMILES' column that properties should be calculated for."
)
parser.add_argument(
    "--output_path",
    "-o",
    type=str,
    help="Path to the output csv file."
)


def calc_properties(output_path: str) -> None:
    # Properties for proteins only.
    EXCLUDE = [
        'protein_weight', 'boman_index', 'charge_density', 'charge', 'instability',
        'aliphaticity', 'hydrophobicity', 'isoelectric_point', 'aromaticity'
    ]

    funcs = {}
    for p in PropertyPredictorRegistry.list_available():
        if p in EXCLUDE:
            continue
        try:
            funcs[p] = PropertyPredictorRegistry.get_property_predictor(p)
        except:
            pass

    mols = pd.read_csv(args.smi_path)
    smiles = mols[['SMILES']]
    if 'Prediction' in mols.columns:
        smiles['IC50'] = mols['Prediction']
    elif 'pred_0' in mols.columns:
        smiles['IC50'] = mols['pred_0']
    for p in funcs:
        try:
            smiles[p] = mols['SMILES'].apply(funcs[p])
        except:
            print(f"[!] Could not calculate property '{p}'")

    smiles.to_csv(output_path, index=False)

if __name__ == '__main__':
    args = parser.parse_args()
    calc_properties(args.output_path)
