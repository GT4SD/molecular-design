"""Lightweight RDKit-backed molecular property helpers."""

from __future__ import annotations

from typing import Dict, Optional

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Parse a SMILES string into an RDKit molecule."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def qed_from_smiles(smiles: str) -> float:
    """Calculate QED for a SMILES string."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return QED.qed(mol)


def compute_rdkit_properties(smiles: str) -> Dict[str, float]:
    """Calculate a small, CI-safe set of RDKit descriptors."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return {
        "molecular_weight": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "qed": QED.qed(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "heavy_atoms": Lipinski.HeavyAtomCount(mol),
    }
