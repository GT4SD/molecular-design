"""Retrosynthesis utilities."""
import logging
import time
from typing import List, Tuple

import numpy as np
from rxn4chemistry import RXN4ChemistryWrapper
from rxn4chemistry.decorators import MININUM_TIMEOUT_BETWEEN_REQUESTS

logger = logging.getLogger('rxn4chemistry-applications:retrosynthesis')

PROCESSING_STATES = set(['NEW', 'PENDING', 'WAITING', 'RUNNING', 'PROCESSING', 'SKIP'])
STOPPING_STATES = set(['SUCCESS', 'ERROR', 'RETROSYNTHESIS_READY', 'DONE'])
AVAILABLE = '#28a30d'
NOT_AVAILABLE = '#990000'


def execute_retrosynthesis(
    rxn4chemistry_wrapper: RXN4ChemistryWrapper,
    smiles: str,
    timeout: int,
    maximum_runtime: int = 3600,
    ai_model: str = "2020-07-01",
    **kwargs,
) -> dict:
    """
    Execute retrosynthesis using a wrapper and given a SMILES and a timeout.
    The kwargs are forwarded to predict_automatic_retrosynthesis of the
    provided wrapper filtering the product key.
    Args:
        rxn4chemistry_wrapper (RXN4ChemistryWrapper): a rxn4tchemistry wrapper.
        smiles (str): a SMILES representing the product.
        timeout (int): timeout in seconds to check the status.
        maximum_runtime (int): maximum runtime in seconds. Defaults to 3600,
            a.k.a., one hour.
        ai_model (str): The identifier for the AI model to use. Defaults to
            "2020-07-01".
    Returns:
        dict: results of the retrosynthesis. In case of problems an empty
        dictionary is returned.
    """
    # making sure the product is not provided twice.
    _ = kwargs.pop('product', None)
    # make sure we don't execute consecutive requests too close in time
    time.sleep(MININUM_TIMEOUT_BETWEEN_REQUESTS)
    try:
        response = rxn4chemistry_wrapper.predict_automatic_retrosynthesis(
            product=smiles, ai_model=ai_model, **kwargs
        )
        prediction_id = response['prediction_id']
    except Exception:
        logger.exception('problem with retrosynthesis submission')
        return {}
    results = {'status': 'NEW'}
    start_time = time.time()
    current_time = time.time()
    not_interrupted = True
    while results['status'] in PROCESSING_STATES and not_interrupted:
        try:
            time.sleep(timeout)
            logger.info(
                'checking retrosynthesis results for prediction {}'.format(
                    prediction_id
                )
            )
            results = rxn4chemistry_wrapper.get_predict_automatic_retrosynthesis_results(  # noqa
                prediction_id
            )
            logger.info('status={}'.format(results['status']))
            current_time = time.time()
            elapsed_time = current_time - start_time
            logger.info('elapsed time: {}'.format(elapsed_time))
            if elapsed_time > maximum_runtime:
                raise RuntimeError('maximum runtime exceeded')
        except Exception:
            not_interrupted = False
            logger.exception('problem with retrosynthesis status check')
    return results if not_interrupted else {}


def is_feasible(tree: dict) -> bool:
    """
    Check whether a tree representing a retrosynthesis sequence is feasible.
    Args:
        tree (dict): tree representing a retrosynthesis sequence.
    Returns:
        bool: feasibility of the retrosynthesis, a.k.a., if all the molecules
        required for the retrosynthesis are available.
    """
    feasibility = []
    if 'borderColor' in tree['metaData']:
        feasibility.append(tree['metaData']['borderColor'] == AVAILABLE)
    for node in tree['children']:
        feasibility.append(is_feasible(node))
    return all(feasibility) if feasibility else False


def get_steps(tree: dict) -> int:
    """
    Get the number of steps from tree representing a retrosynthesis sequence.
    Args:
        tree (dict): tree representing a retrosynthesis sequence.
    Returns:
        int: number of steps.
    """
    steps = 0
    if 'children' in tree and len(tree['children']):
        children_steps = [get_steps(node) for node in tree['children']]
        steps = 1 + max(children_steps)
    return steps


def get_num_reactants(tree: dict) -> int:
    """
    Get the number of reactants from tree representing a retrosynthesis sequence.
    Args:
        tree (dict): tree representing a retrosynthesis sequence.
    Returns:
        int: number of reactants.
    """
    reactants = 0
    if 'children' in tree and len(tree['children']):
        reactants = sum([get_num_reactants(node) for node in tree['children']])
    else:
        reactants = 1
    return reactants


def get_num_reactions(tree: dict) -> int:
    """
    Get the number of reactions from tree representing a retrosynthesis sequence.
    Args:
        tree (dict): tree representing a retrosynthesis sequence.
    Returns:
        int: number of reactions.
    """
    reactions = 0
    if 'children' in tree and len(tree['children']):
        reactions = 1 + sum([get_num_reactions(node) for node in tree['children']])
    return reactions


def get_all_smiles(tree: dict) -> List[str]:
    """
    Get list of SMILES from all levels of reaction tree
    Args:
        tree (dict): tree representing a retrosynthesis sequence.
    Returns:
        List[str]: List of SMILES that are part of the reaction tree.
    """
    reactions = []
    if 'children' in tree and len(tree['children']):
        reactions.append([node['smiles'] for node in tree['children']])
    for node in tree['children']:
        reactions.extend(get_all_smiles(node))
    return reactions


def get_rxn_smiles(tree: dict) -> List[str]:
    """
    Get list of reaction SMILES from all levels of reaction tree
    Args:
        tree (dict): tree representing a retrosynthesis sequence.
    Returns:
        List[str]: List of reaction SMILES that are part of the reaction tree.
    """
    reactions = []
    if 'children' in tree and len(tree['children']):
        reactions.append(
            '{}>>{}'.format(
                '.'.join([node['smiles'] for node in tree['children']]),
                tree['smiles']
            )
        )
    for node in tree['children']:
        reactions.extend(get_rxn_smiles(node))
    return reactions


def get_reaction_smiles_depths(rxnsmis: List[str], steps: int) -> List[int]:
    """
    Get depths of reactants from reaction SMILES
    Args:
        rxnsmis (List[str]): List of reaction SMILES
        steps (int): Number of steps for the synthesis (i.e. tree depth)
    Returns:
        List[int]: depths of reactants. List of equal length to rxnsmis.
    """

    # If number of reactions is identical to tree depth, we can shortcut
    if len(rxnsmis) == steps:
        return list(range(steps))
    else:
        # Otherwise, we need to calculate the depths
        return get_depths(rxnsmis)


def get_depths(rxnsmis: List[str]) -> List[int]:
    """
    Finds the depth of each node in the synthesis based on list of reaction SMILES.
    Args:
        rxnsmis (List[str]): List of reaction SMILES strings.
    Returns:
        List[int]: List of depths, one for each reaction SMILES string.
    """

    # Assuming that first element is the root node
    depths = [0]

    # Dictionary to store precursors and products on each level
    level_dict = {0: dict(zip(["precursors", "products"], parse_rxnsmi(rxnsmis[0])))}
    for rxnsmi in rxnsmis[1:]:
        # Parse reaction SMILES string
        precursors, products = parse_rxnsmi(rxnsmi)
        # Check whether node is child of one of the preceding levels
        found = False
        for level in range(1 + np.max(depths))[::-1]:
            if any([p in level_dict[level]["precursors"] for p in products]):
                # Found a child of that level
                lev = level + 1
                if lev in depths:
                    level_dict[lev]["precursors"].extend(precursors)
                    level_dict[lev]["products"].extend(products)
                else:
                    level_dict[lev] = dict(
                        zip(["precursors", "products"], [precursors, products])
                    )
                found = True
                depths.append(lev)
                break
        if not found:
            # In case no product appears as precursor in previous reactions we
            # just assume that this reaction is a child of previous reaction.
            lev = np.max(depths) + 1
            depths.append(lev)
            level_dict[lev] = dict(
                zip(["precursors", "products"], [precursors, products])
            )
    return depths


def parse_rxnsmi(rxnsmi: str) -> Tuple[List[str], List[str]]:
    """
    Parses a reaction SMILES string into a list of precursors and products.
    Args:
        rxnsmi (str): Reaction SMILES string.
    Returns:
        Tuple[List[str],List[str]]: A tuple of lists of precursors and products.
    """
    # Split reaction SMILES string into precursors and products
    rxnsmi_split = rxnsmi.split(">>")

    precursors = rxnsmi_split[0].split(".")
    products = rxnsmi_split[1].split(".")
    return precursors, products