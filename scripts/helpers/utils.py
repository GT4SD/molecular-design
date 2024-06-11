import os
import subprocess as sp
import re
import sys

import numpy as np
import psutil
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

UPPER = 11
LOWER = 1
TANI_DICT = {}


def nanomolar_to_pic50(x):
    return max(min(-np.log10(x * 1e-9), UPPER), LOWER)


def molar_to_pic50(x):
    return max(min(-np.log10(x), UPPER), LOWER)


def str2float(string_value: str) -> float:
    """
    Convert a string to a float.

    Args:
        string_value (str): string value representing a float.

    Returns:
        float: the converted value.
    """
    string_value = str(string_value)
    is_number = re.compile(r"([<>])?\s*([\d+|\d+\.\d+]+)")
    qualifier, number = is_number.search(string_value).groups()
    number = float(number)
    number += sys.float_info.min * (1 if qualifier == ">" else -1) if qualifier else 0.0
    return number


def get_fast_tani(mol1: Chem.Mol, mol2: Chem.Mol, radius: int = 2):
    # Assumes mol1 and mol2 are rdkit.Chem.Mol objects, no checks, good to screen unique DBs
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=radius)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=radius)
    return DataStructs.FingerprintSimilarity(fp1, fp2)


def get_tani(smi1: str, smi2: str, radius: int = 2):
    if f"{smi1}_{smi2}" not in TANI_DICT.keys():
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)

        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=radius)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=radius)

        TANI_DICT[f"{smi1}_{smi2}"] = DataStructs.FingerprintSimilarity(fp1, fp2)

    return TANI_DICT[f"{smi1}_{smi2}"]


def algebraic_tanimoto(fp1, fp2):
    nominator = np.dot(fp1, fp2)
    denominator = np.dot(fp1, fp1) + np.dot(fp2, fp2) - np.dot(fp1, fp2)
    return nominator / denominator


def cuda():
    return torch.cuda.is_available()


def get_gpu_memory():
    if not cuda():
        return 0, 0, 0
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

    tot_m, used_m, free_m = map(int, os.popen("free -t -m").readlines()[-1].split()[1:])
    return memory_free_values, used_m, tot_m


def get_cpu_memory():
    mem = psutil.virtual_memory()
    return mem.total / 1000**3, mem.percent, psutil.cpu_percent()


def get_process_mmeory():
    process = psutil.Process(os.getpid())
    return process.memory_percent()
