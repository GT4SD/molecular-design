import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from molecule_generation.chem.motif_utils import fragment_into_candidate_motifs
from pytoda.smiles.processing import tokenize_smiles
from pytoda.smiles.transforms import Canonicalization, Kekulize
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("moler_generator")
logger.setLevel(level=logging.INFO)

hs_explicit = Kekulize(all_hs_explicit=True)
kekulize = Kekulize()
can = Canonicalization()

DECORATION_PIECES = [
    "C([CH3])",
    "C([CH2]([CH3]))",
    "C([CH]([CH3])([CH3]))",
    "C([CH2][CH]([CH3])([CH3]))",
    "C([C]([CH3])([CH3])([CH3]))",
]
CANONICAL_DECORATIONS = list(map(lambda x: "(" + can(x)[1:] + ")", DECORATION_PIECES))
"""
For a SMILES-based RT, this approach works to facilitate a decoration where the seed molecule
is kept identical but some methyl groups are removed and replaced by larger motifs.


 SELFIES models such decorations are not so easy. While the connection points
can easily be detected in SMILES, the molecule's ring structure can easily be
broken by the model adding a `[Branch1]` token.
In SMILES instead, if the model predicts a ring symbol (`1`) the molecule becomes
invalid and is directly filtered away.

SOLUTION:

"""


def substruct_match(
    query_smiles: str,
    ref_mols: List,
    query_in_ref: bool = False,
    ref_in_query: bool = True,
) -> bool:
    """
    Test whether a query molecule is a substructure of any molecule in a reference list.

    Args:
        query_smiles: The SMILES of interest
        ref_mols: A list of reference molecules against which the query is checked
        query_in_ref: If True, checks whether the query is a substructure of any molecule in the
            reference list.
        ref_in_query: If True, checks whether any molecule in the reference list is a substructure
            of the query.
    Returns:
        Whether a match was found.

    """
    query_mol = Chem.MolFromSmiles(query_smiles)
    if not query_in_ref and not ref_in_query:
        raise ValueError("No checking performed if both bools are False")

    if query_in_ref and not ref_in_query:
        match = any([m.HasSubstructMatch(query_mol) for m in ref_mols])
    elif not query_in_ref and ref_in_query:
        match = any([query_mol.HasSubstructMatch(m) for m in ref_mols])
    else:
        match_first = any([m.HasSubstructMatch(query_mol) for m in ref_mols])
        match_second = any([query_mol.HasSubstructMatch(m) for m in ref_mols])
        match = match_first or match_second
    return match


def decorate_smiles(
    smi: str, keep_prob: float = 0.5, max_length: int = 6, rt_language: str = "SELFIES"
) -> str:
    """
    This function stochastically decorates a seed SMILES by replacing
    each methyl group with one of the five decoration pieces above.

    Args:
        smi: SMILES to be decorated.
        keep_prob: Probability to keep each methyl group. Defaults to 0.5.

    Returns:
        The decorated SMILES
    """
    if rt_language not in ["SMILES", "SELFIES"]:
        raise ValueError(f"Currently no support for RT language {rt_language}")

    hs_smi = hs_explicit(smi)
    toks = tokenize_smiles(hs_smi)
    idxs = [i for i, t in enumerate(toks) if t == "[CH]"]
    new = hs_smi
    counter = 0
    while hs_smi == new and counter < 50:
        new_toks = []
        for i, t in enumerate(toks):
            adapt = np.random.uniform(low=0, high=1) > keep_prob
            if adapt and i > 0 and i + 1 < len(toks) and i in idxs:
                if rt_language == "SMILES":
                    attachment = np.random.choice(DECORATION_PIECES)
                    new_toks.append(attachment)
                elif rt_language == "SELFIES":
                    attachment = np.random.randint(low=1, high=max_length)
                    # Assuming that '[Se]' is a rare element so we abuse it for this task
                    new_toks.append("C(" + "[Se]" * attachment + ")")
            else:
                new_toks.append(t)
        new = "".join(new_toks)
        counter += 1
    if new == hs_smi:
        logger.warning(f"Could not decorate {smi}")
    return kekulize(new)


def parse_rt_samples(samples: Tuple[str]) -> List[Dict[str, Any]]:
    """
    Convert a tuple of RT samples to a list of dicts that can be passed
    to pd.DataFrame.

    Args:
        samples: RT generated sample

    Returns:
        List of dicsts
    """

    # Create properties
    properties = [s.split("<")[1] for s in samples[0][1].split(">")[:-1]]
    result = []
    # Fill properties
    for sample in samples:
        result.append({"SMILES": sample[0]})
        for prop in properties:
            value = float(sample[1].split(prop)[-1][1:].split("<")[0])
            result[-1].update({prop: value})
    return result


def moler_motif_extractor(smi: str, cut_leaf_edges: bool = False) -> List[str]:
    """
    Given a SMILES, use the `fragment_into_candidate_motifs` method
    from MoLeR to extract motifs.

    Args:
        smi: SMILES to extract motifs from
        cut_leaf_edges: whether molecules are cut on leaf edges

    Returns:
        List of motifs, given as SMILES
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []

    frags_with_ann = fragment_into_candidate_motifs(mol, cut_leaf_edges=cut_leaf_edges)
    frag_smis = [Chem.MolToSmiles(f[0]) for f in frags_with_ann]
    # Remove invalid motifs and flatten motifs if they contain a dot
    frag_smis = ".".join([f for f in frag_smis if f is not None]).split(".")
    return frag_smis


def scaffold_extractor(smi: str) -> List[str]:
    """
    Given a SMILES, use the `MurckoScaffold.GetScaffoldForMol` method
    to extract the murcko scaffold

    Args:
        smi: SMILES to extract motifs from

    Returns:
        List of string, with only one item, the scaffold
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []

    scaff = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    return scaff.split(".")


def shredding():
    # TODO: Wrapper for shredding mechanism in libpqr
    pass


def extract_motifs_from_smiles_list(
    smiles: List[str],
    extractor_string: str,
    labels: Optional[List[float]] = None,
    filter_atoms: List[str] = [],
) -> Tuple[str, List[float]]:
    """
    Given a list of SMILES this function extracts all motifs with at least
        min_size atoms and returns a list of such motifs together
        with their relative frequencies.
    Args:
        smiles: A list of SMILES.
        extractor_string: A key for the `MOTIF_FACTORY`
        labels: Optional list of floats. If not provided, relative frequencies
            are computed.
        filter_atoms: A list of atoms that cannot be considered (motifs that encounter
            it are removed).

    Returns:
        A list of mol objects.
        A list of scores, one for each motif. If labels are given, this is the
            average of all labels which contain a motif. If no labels are given
            this is the relative frequency of the motif.
    """
    motif_dict = defaultdict(list)
    label_mode = labels is not None
    if label_mode:
        assert len(labels) == len(
            smiles
        ), f"Provide one label per SMILES string, {len(labels)}, {len(smiles)}"
    else:
        labels = [1] * len(smiles)
    for smi, label in tqdm(
        zip(smiles, labels),
        desc=f"Extracing motifs: {extractor_string}",
        total=len(smiles),
    ):
        motifs = MOTIF_FACTORY[extractor_string](smi)
        for motif in motifs:
            if any([x in motif for x in filter_atoms]) or motif == "":
                continue
            motif_dict[motif].append(label)

    # Get final scores
    for motif in motif_dict.keys():
        if label_mode:
            motif_dict[motif] = np.mean(motif_dict[motif])
        else:
            motif_dict[motif] = len(motif_dict[motif]) / len(smiles)
    return list(motif_dict.keys()), np.array(list(motif_dict.values()))


MOTIF_FACTORY = {
    "moler": moler_motif_extractor,
    "libpqr": shredding,
    "scaffold": scaffold_extractor,
}
