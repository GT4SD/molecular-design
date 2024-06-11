import json
import logging
import os
import sys

import torch
from toxsmi.models import MODEL_FACTORY
from paccmann_predictor.utils.utils import get_device
from pytoda.smiles.smiles_language import SMILESTokenizer
from pytoda.datasets import AnnotatedDataset, SMILESTokenizerDataset

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("bimca_utils")
logger.setLevel(logging.INFO)


def restore_toxsmi(checkpoint_path: str) -> MODEL_FACTORY["mca"]:
    """
    Restores a `Toxsmi` from a checkpoint path. Assumes
    that the checkpoint_path points to `model_path/xyz.ckpt`
    and that `model_path` also contains a parameter file
    and a SMILES language.

    Args:
        checkpoint_path: Path to checkpoint

    Raises:
        NameError: _description_

    Returns:
        AffinityModel (BiMCA) object in eval mode
    """

    model_folder = os.path.dirname(os.path.dirname(checkpoint_path))

    with open(os.path.join(model_folder, "model_params.json")) as fp:
        params = json.load(fp)
    # setting device
    device = get_device()
    smiles_language = SMILESTokenizer(
        vocab_file=os.path.join(model_folder, "smiles_language"),
        padding_length=params.get("padding_length", None),
        randomize=False,
        add_start_and_stop=params.get("start_stop_token", True),
        padding=params.get("padding", True),
        augment=False,
        canonical=params.get("test_canonical", params.get("augment_smiles", False)),
        kekulize=params.get("kekulize", False),
        all_bonds_explicit=params.get("bonds_explicit", False),
        all_hs_explicit=params.get("all_hs_explicit", False),
        remove_bonddir=params.get("remove_bonddir", False),
        remove_chirality=params.get("remove_chirality", False),
        selfies=params.get("selfies", False),
        sanitize=params.get("sanitize", False),
    )
    smiles_language.set_smiles_transforms(augment=False)

    if not os.path.isfile(checkpoint_path):
        raise TypeError(f"{checkpoint_path} is not a `.ckpt` file.")
    try:
        # ckpt = torch.load(checkpoint_path, map_location=device)
        model = MODEL_FACTORY["mca"](params).to(device)
        # model.load_state_dict(ckpt)
        model.load(checkpoint_path, map_location=device)
        logger.info(f"Found and restored existing model {checkpoint_path}")
    except NameError:
        raise NameError("Could not restore model")
    model.eval()
    return model, smiles_language


def get_loader(
    label_path: str,
    smiles_path: str,
    smiles_language: SMILESTokenizer,
    batch_size: int = 512,
) -> torch.utils.data.DataLoader:
    # Assemble datasets
    smiles_dataset = SMILESTokenizerDataset(
        smiles_path, smiles_language=smiles_language
    )
    dataset = AnnotatedDataset(
        annotations_filepath=label_path, dataset=smiles_dataset, label_columns=["Label"]
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return dataset, loader
