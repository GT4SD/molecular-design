import os
import sys
import json
import logging
import argparse
import pandas as pd
from rxn4chemistry import RXN4ChemistryWrapper
from helpers.rxn_analyser import execute_retrosynthesis

# logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("rxn-retrosynthesis")

# define the parser arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "input_filepath", type=str, help="path to the .csv containig the molecules."
)
parser.add_argument(
    "-a", "--api_key", type=str, help="API key to access IBM RXN.", required=True
)
parser.add_argument(
    "-p",
    "--project_id",
    type=str,
    help="project identifier from IBM RXN.",
    required=True,
)
parser.add_argument(
    "-n",
    "--name",
    type=str,
    help="Name of project",
    required=False,
    default="BatchRetro",
)

parser.add_argument(
    "-s",
    "--steps",
    type=int,
    help="max retrosynthesis steps. Defaults to 6.",
    default=6,
    required=False,
)
parser.add_argument(
    "-b",
    "--beams",
    type=int,
    help="number of beams for the search. Defaults to 10.",
    default=10,
    required=False,
)
parser.add_argument(
    "-t",
    "--timeout",
    type=int,
    help="timeout in seconds to check the status. Default to 60.",
    default=60,
    required=False,
)


def retrosynthesis_analysis(
    input_filepath: str,
    api_key: str,
    project_id: str,
    name: str,
    max_steps: int,
    nbeams: int,
    timeout: int,
):
    """
    Retrosynthesis analysis with a given timeout.
    The analysis will generate a report in a folder called results inside
    the folder containing the .csv file.
    The report consist of a .json file for each molecules containing the
    results retrieved by rxn4chemistry for an automatic retrosynthesis.
    Args:
        input_filepath (str): path to the .csv containig the molecules
            in a SMILES column.
        api_key (str): API key to access IBM RXN.
        project_id (str): project identifier from IBM RXN.
        max_steps (int): max retrosynthesis steps.
        nbeams (int): number of beams for the search.
        timeout (int): timeout in seconds to check the status.
    """
    # setup the client
    rxn4chemistry_wrapper = RXN4ChemistryWrapper(api_key=api_key)
    rxn4chemistry_wrapper.set_project(project_id)
    # get data information and setup resuls folder
    path = os.path.dirname(input_filepath)
    set_name = os.path.splitext(os.path.basename(input_filepath))[0]
    results_path = os.path.join(path, "results", set_name)
    os.makedirs(results_path, exist_ok=True)
    # read the data
    df = pd.read_csv(input_filepath)
    logger.info("processing {} molecules".format(df.shape[0]))
    for index, row in df.iterrows():
        if "ID" in row:
            identifier = f"{name}_{row.ID}"
        else:
            identifier = f"{name}_{index}"
        smiles = row["SMILES"]
        output_filepath = os.path.join(results_path, "{}.json".format(identifier))
        if os.path.exists(output_filepath):
            logger.info(
                "retrosynthesis already present for {} with SMILES={}".format(
                    identifier, smiles
                )
            )
            continue
        logger.info(
            "starting retrosynthesis of {} with SMILES={}".format(identifier, smiles)
        )
        results = execute_retrosynthesis(
            rxn4chemistry_wrapper=rxn4chemistry_wrapper,
            smiles=smiles,
            timeout=timeout,
            max_steps=max_steps,
            nbeams=nbeams,
        )
        if not results:
            logger.error(
                "retrosynthesis problem for {} with SMILES={}".format(
                    identifier, smiles
                )
            )
            continue
        logger.info(
            "completed retrosynthesis of {} with SMILES={}".format(identifier, smiles)
        )
        with open(output_filepath, "w") as fp:
            json.dump(results, fp)
        logger.info("retrosynthesis stored: {}".format(output_filepath))


if __name__ == "__main__":
    # parse arguments
    args = parser.parse_args()
    # run the analysis
    retrosynthesis_analysis(
        input_filepath=args.input_filepath,
        api_key=args.api_key,
        project_id=args.project_id,
        name=args.name,
        max_steps=args.steps,
        nbeams=args.beams,
        timeout=args.timeout,
    )