import os
import uuid
from argparse import ArgumentParser, Namespace

import imageio
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from torch.nn import L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..util_methods import *
from . import build_satmae_finetune, build_satmae_temporal_finetune

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# sorry couldn't figure out a better way to do this
SATMAE_PATHS = [
    (None, 1),
    (None, 2),
    (None, 3),
    (None, 4),
    (None, 5),
    ("/data/output/52518326-c", 1),
    ("/data/output/c125a540-5", 2),
    ("/data/output/03f0e579-b", 3),
    ("/data/output/6a1fce5c-b", 4),
    ("/data/output/fa681052-0", 5),
    # ("/data/output/58b9ae03-b", 1),
    # ("/data/output/c125a540-5", 2),
    # ("/data/output/03f0e579-b", 3),
]

assert torch.cuda.is_available(), "Using GPU is strongly recommended"
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")


class ViTForRegression(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        # Assuming the original model outputs 768 features from the transformer
        self.regression_head = nn.Linear(1024, num_classes)  # Output one continuous variable

    def forward(self, *args):
        outputs = self.base_model(*args)
        # We use the last hidden state
        return torch.sigmoid(self.regression_head(outputs))


def main(args, fold, name, outdir=None, model=None, loaders=None):
    print(name, fold)

    if outdir is None:
        outdir = os.path.join(args.output_path, str(uuid.uuid4())[:10])
    os.makedirs(outdir, exist_ok=True)
    print("Output dir:", outdir)
    print("Fold:", fold)

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(outdir)
    else:
        print("Tensorboard not available: not logging progress")

    predict_target = ["deprived_sev"]

    if loaders is None:
        if args.temporal:
            train_dataset, _ = get_datasets(
                f"dhs_centroids_with_pov_scaled.csv",
                args.imagery_path,
                predict_target,
                split=False,
                temporal=args.temporal,
                train=True,
                landsat=args.landsat,
            )
            test_dataset, _ = get_datasets(
                f"dhs_centroids_with_pov_scaled.csv",
                args.imagery_path,
                predict_target,
                split=False,
                temporal=args.temporal,
                train=False,
                landsat=args.landsat,
            )
        else:
            train_dataset, _ = get_datasets(
                f"data/train_fold_{fold}.csv",
                args.imagery_path,
                predict_target,
                split=False,
                temporal=args.temporal,
                train=True,
                landsat=args.landsat,
            )
            test_dataset, _ = get_datasets(
                f"data/test_fold_{fold}.csv",
                args.imagery_path,
                predict_target,
                split=False,
                temporal=args.temporal,
                train=False,
                landsat=args.landsat,
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            persistent_workers=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            persistent_workers=True,
        )

    else:
        print("Using existing dataloaders")
        train_loader, test_loader = loaders

    build_fn = build_satmae_temporal_finetune if args.temporal else build_satmae_finetune

    if model is not None:
        print("Model provided, not loading checkpoints")
    elif SATMAE_PATHS[fold - 1][0] is not None:
        base_model = build_fn(
            Namespace(
                num_classes=0,
                drop_path=0.1,
                global_pool=False,
                satmae_type="vit_large_patch16",
                pretrained_model=None,
            )
        )

        model = base_model

        ckpt = torch.load(os.path.join(SATMAE_PATHS[fold - 1][0], "model_2020_best_nl.pth"))
        model_ckpt = {
            k.replace("base_model.", ""): v
            for k, v in ckpt["model_state_dict"].items()
            if "head" not in k
        }
        model.load_state_dict(model_ckpt, strict=False)
        for p in model.parameters():
            p.requires_grad_(False)

    else:
        base_model = build_fn(
            Namespace(
                num_classes=0,
                drop_path=0.1,
                global_pool=False,
                satmae_type="vit_large_patch16",
                pretrained_model=(
                    "/home/jupyter/ckpts/pretrain_fmow_temporal.pth"
                    if args.temporal
                    else "/home/jupyter/ckpts/fmow_pretrain.pth"
                ),
            )
        )

        model = base_model

    # Move model to appropriate device
    model = model.to(device)

    model.eval()

    def extract(dataloader, filename):
        features = []
        for batch in tqdm(dataloader, ncols=100):
            if args.temporal:
                images, ts, target = batch
                images, ts = images.to(device), ts.to(device)
                with torch.no_grad():
                    outputs = model(images, ts)
                    outputs = outputs.cpu().numpy()

            else:
                images, target = batch
                images = images.to(device)

                # Forward pass
                with torch.no_grad():
                    outputs = model(images)
                    outputs = outputs.cpu().numpy()

            # print(outputs.shape)
            features.append(np.concatenate([outputs, target], 1))
            assert len(features[-1].shape) == 2 and features[-1].shape[-1] == 1025

        features = np.concatenate(features, 0)
        np.save(os.path.join(outdir, filename), features)

    # Validation phase
    extract(train_loader, f"train_{name}.npy")
    extract(test_loader, f"test_{name}.npy")
    return model, (train_loader, test_loader)


if __name__ == "__main__":
    parser = ArgumentParser(description="Finetune SatMAE")
    parser.add_argument("--imagery_path", type=str, default="/data/esa_10")
    parser.add_argument("--output_path", type=str, default="/data/output")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sentinel", action="store_false", dest="landsat")
    parser.add_argument("--temporal", action="store_true")
    args = parser.parse_args()
    outdir = os.path.join(args.output_path, str(uuid.uuid4())[:10])
    models = {}
    loaders = {}
    for mid, lid in SATMAE_PATHS:
        m, l = main(
            args,
            lid,
            ("raw_" if mid is None else "finetuned_") + str(lid),
            outdir,
            models[mid] if mid in models else None,
            loaders[lid] if lid in loaders else None,
        )
        models[mid] = m
        loaders[lid] = l
