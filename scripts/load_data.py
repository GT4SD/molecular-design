import requests
import os
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from helpers import utils


parser = argparse.ArgumentParser()
parser.add_argument(
    "--uniprot",
    "-u",
    type=str,
    help="UniProt code for protein to fetch."
)
parser.add_argument(
    "--output_dir",
    "-o",
    default="data/",
    type=str,
    help="The directory where datasets are saved."
)
parser.add_argument(
    "--affinity_cutoff",
    type=int,
    default=10_000,
    help="Affinity cutoff point."
)
parser.add_argument(
    "--affinity_type",
    type=str,
    default='IC50',
    choices=['IC50', 'Kd'],
    help="IC50 or Kd"
)
parser.add_argument(
    "--train_size",
    type=float,
    default=0.8,
    help="Proportion of dataset included in the train split."
)
parser.add_argument(
    "--binary_labels",
    action='store_true',
    help="Enable binary classification. If not specified, the default mode is regression."
)
def fetch(
    uniprot: str,
    affinity_cutoff: int,
    affinity_type: str,
) -> pd.DataFrame:
    url = f"https://bindingdb.org/axis2/services/BDBService/getLigandsByUniprots?uniprot={uniprot}&cutoff={affinity_cutoff}&response=application/json"
    response = requests.get(url)
    assert response.status_code == 200, '[x] Failed to fetch data from bindingdb'

    data = response.json()
    affinities = data['getLigandsByUniprotsResponse']['affinities']
    df = pd.DataFrame(affinities)
    df = df[df['affinity_type'] == affinity_type]
    df = df[['smile', 'monomerid', 'affinity']]
    df['affinity'] = df['affinity'].apply(utils.str2float)
    df['affinity'] = df['affinity'].apply(utils.nanomolar_to_pic50)
    assert df['smile'].nunique() == df['monomerid'].nunique(), 'ID is not unique'
    df = df[~df['smile'].duplicated(keep=False)]
    df = df.reset_index(drop=True)
    return df


if __name__ == '__main__':
    args = parser.parse_args()
    dataset = fetch(args.uniprot, args.affinity_cutoff, args.affinity_type)

    # three files. mols.smi list of all the smiles. Then we have train.csv and val.csv
    mol_path = os.path.join(args.output_dir, 'mols.smi')
    train_path = os.path.join(args.output_dir, 'train.csv')
    val_path = os.path.join(args.output_dir, 'valid.csv')
    # Save smiles and id without header. Note that this dataset uses tab delimiter.
    dataset[['smile', 'monomerid']].to_csv(mol_path, index=False, header=False, sep='\t')
    # Training dataset have columns Label,sampling_frequency,mol_id
    dataset = dataset.rename(columns={'affinity':'Label', 'monomerid':'mol_id'})
    dataset['sampling_frequency'] = 'high'
    dataset = dataset[['Label', 'sampling_frequency', 'mol_id']]

    if args.binary_labels:
        dataset['Label'] = dataset['Label'].apply(lambda x: 1 if x > 6 else 0)
    train, validation = train_test_split(dataset, train_size=args.train_size, random_state=1911)
    train.to_csv(train_path, index=False, header=True)
    validation.to_csv(val_path, index=False, header=True)
