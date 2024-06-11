import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_path",
    "-i",
    type=str,
    help="Path to the molecule csv file to run toxsmi inference on."
)

def create_datasets(input_path: str) -> None:
    """
    Given the generated molecules from RT in format
        SMILES,qed,seed,to_mask,decorate
    turn it into
        SMILES\tmol_id

    then for each mol_id generate a fake dataset
        Label,mol_id
    """
    df = pd.read_csv(input_path)
    df = df[['SMILES']]
    df['mol_id'] = pd.factorize(df['SMILES'])[0]

    df.to_csv('generated.smi', index=False, header=False, sep='\t')

    n = df['mol_id'].max()
    df_fake = pd.DataFrame({'Label': [0.0] * n, 'mol_id': range(n)})
    df_fake = df_fake[['Label', 'mol_id']]
    df_fake.to_csv('dummy_data.csv', index=False)

if __name__ == '__main__':
    args = parser.parse_args()
    create_datasets(args.input_path)
