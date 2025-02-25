import os
import uuid
from argparse import ArgumentParser, Namespace

import imageio
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from torch.nn import L1Loss, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from ..util_methods import *
from . import build_satmae_finetune, build_satmae_temporal_finetune

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


assert torch.cuda.is_available(), "Using GPU is strongly recommended"
device = torch.device("cuda")


def main(args):
    outdir = os.path.join(args.output_path, str(uuid.uuid4())[:10])
    os.makedirs(outdir, exist_ok=True)
    print("Output dir:", outdir)
    print("Train data:", args.dhs_path)

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(outdir)
    else:
        print("Tensorboard not available: not logging progress")

    predict_target = [
        "h10",
        "h3",
        "h31",
        "h5",
        "h7",
        "h9",
        "hc70",
        "hv109",
        "hv121",
        "hv106",
        "hv201",
        "hv204",
        "hv205",
        "hv216",
        "hv225",
        "hv271",
        "v312",
    ]
    train_dataset, val_dataset, num_classes = get_datasets(
        args.dhs_path,
        args.imagery_path,
        predict_target,
        temporal=args.temporal,
        landsat=args.landsat,
        img_size=args.img_size,
    )

    # Set your desired seed
    set_seed(args.random_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    model_args = Namespace(
        num_classes=99,
        drop_path=0.1,
        global_pool=False,
        satmae_type="vit_large_patch16",
        pretrained_model=args.pretrained_ckpt,
        img_size=args.img_size,
    )
    if args.temporal:
        base_model = build_satmae_temporal_finetune(model_args)
    else:
        base_model = build_satmae_finetune(model_args)  

    model = base_model.to(device)

    # Setup the optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5, weight_decay=1e-6)
    if args.loss == "l1":
        loss_fn = L1Loss()
    elif args.loss == "l2":
        loss_fn = MSELoss()
    else:
        raise ValueError("Loss function other than 'l1' or 'l2' not supported")

    best_error = np.inf
    curr_patience = args.stopping_patience

    if args.enable_profiling:
        timer_start = torch.cuda.Event(enable_timing=True)
        timer_end = torch.cuda.Event(enable_timing=True)

    iteration = 0

    stopping = False
    for epoch in range(args.epochs):
        model.train()
        print("Training...")
        if args.enable_profiling:
            timer_start.record()

        for batch in tqdm(train_loader):
            images, targets = batch
            images, targets = images.to(device), targets.to(device)

            if args.enable_profiling:
                timer_end.record()
                torch.cuda.synchronize()
                loading_time = timer_start.elapsed_time(timer_end)
                timer_start.record()

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.enable_profiling:
                timer_end.record()
                torch.cuda.synchronize()
                step_time = timer_start.elapsed_time(timer_end)

            if tb_writer and iteration % 5 == 0:
                tb_writer.add_scalar("loss", loss.item(), iteration)
                if args.enable_profiling:
                    tb_writer.add_scalar("timer/load", loading_time, iteration)
                    tb_writer.add_scalar("timer/step", step_time, iteration)
            # break

            iteration += 1

        # Validation phase
        model.eval()
        val_loss = []
        indiv_loss = []
        print("Validating...")
        for batch in tqdm(val_loader):
            images, targets = batch
            images, targets = images.to(device), targets.to(device)

            # Forward pass
            with torch.no_grad():
                outputs = model(images)
            batch_loss = loss_fn(outputs, targets)
            val_loss.append(batch_loss.item())
            indiv_loss.append(torch.mean(torch.abs(outputs - targets), axis=0))
            break

        # Compute mean validation loss
        mean_val_loss = np.mean(val_loss)
        mean_indiv_loss = torch.stack(indiv_loss).mean(dim=0)

        print("delta:", np.abs(mean_val_loss - best_error))
        if best_error - mean_val_loss < args.stopping_delta:
            curr_patience -= 1
            print("Patience:", curr_patience)
            if curr_patience == 0:
                stopping = True
        else:
            curr_patience = args.stopping_patience

        if mean_val_loss < best_error:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                mean_val_loss,
                filename=os.path.join(outdir, "model_2020_best_nl.pth"),
            )
            best_error = mean_val_loss
        print("Output dir:", outdir)
        print(
            f"Epoch [{epoch+1}/{args.epochs}], Validation Loss: {mean_val_loss}, Individual Loss: {mean_indiv_loss}"
        )

        if tb_writer:
            tb_writer.add_scalar("val_loss", mean_val_loss, iteration)
        save_checkpoint(
            model,
            optimizer,
            epoch,
            mean_val_loss,
            filename=os.path.join(outdir, "model_2020_last_nl.pth"),
        )

        if stopping:
            break
        # break


if __name__ == "__main__":
    parser = ArgumentParser(description="Finetune SatMAE")
    parser.add_argument("--imagery_path", type=str, default="/data/esa_10")
    parser.add_argument("--dhs_path", type=str)
    parser.add_argument("--output_path", type=str, default="/data/output")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--sentinel", action="store_false", dest="landsat")
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--stopping_delta", type=float, default=5e-4)
    parser.add_argument("--stopping_patience", type=int, default=5)
    parser.add_argument("--loss", type=str, default="l1")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--pretrained_ckpt", type=str, default="")
    parser.add_argument("--enable_profiling", action="store_true")
    parser.add_argument("--img_size", type=int, default=224)
    main(parser.parse_args())
