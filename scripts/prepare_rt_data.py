import pandas as pd
import argparse
from gt4sd.properties import PropertyPredictorRegistry

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smi_path",
    "-s",
    type=str,
    help="Path to generated SMILES csv file."
)
parser.add_argument(
    "--output_path",
    "-o",
    type=str,
    help="Path to the output csv file."
)

def calc_properties(output_path: str) -> None:
    qed = PropertyPredictorRegistry.get_property_predictor('qed')
    df = pd.read_csv(args.smi_path)
    df['<qed>'] = df['SMILES'].apply(qed)
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    args = parser.parse_args()
    calc_properties(args.output_path)
