"""
Training script for LBA binding affinity prediction with UnetHeteroGVPForLBA.

Usage:
    python scripts/train_lba.py [options]

Key options (all have defaults):
    --split       30 or 60   (sequence-identity split, default 30)
    --epochs      int        (default 100)
    --batch_size  int        (default 8)
    --lr          float      (default 5e-4)
    --wd          float      (weight decay, default 1e-3)
    --s_dim       int        (default 128)
    --num_layers  int        (default 5)
    --pool        str        (sum|mean, default sum)
    --num_workers int        (default 4)
    --devices     int        (number of GPUs, default 1)
    --name        str        (run name for logging, default auto)
    --wandb                  (enable W&B logging, default: CSV only)
    --no_test                (skip test after training)
    --ckpt_path   str        (resume from checkpoint)
"""
import argparse
import os
import sys

sys.path.insert(0, "/scratch/ziyiz14/data/GNN_UNet")

import torch
torch.set_float32_matmul_precision("high")   # avoids precision warnings; stable on RTX

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torchmetrics import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef

from proteinworkshop.datasets.lba_hetero_dataset import LBAHeteroDataModule
from proteinworkshop.models.graph_encoders.unet_hetero_gvp import UnetHeteroGVPForLBA, UnetHeteroGVPEncoderOnlyForLBA


# ------------------------------------------------------------------ #
# Lightning Module                                                      #
# ------------------------------------------------------------------ #

class LBALightningModule(L.LightningModule):
    """Wraps UnetHeteroGVPForLBA with training/val/test steps and metrics."""

    def __init__(
        self,
        # encoder
        model_name: str="UnetHeteroGVPForLBA",
        s_dim: int = 128,
        v_dim: int = 16,
        s_dim_edge: int = 32,
        v_dim_edge: int = 1,
        r_max: float = 10.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        num_layers: int = 5,
        pool: str = "sum",
        fps_ratio: float = 0.6,
        cross_cutoff: float = 6.0,
        enc_drop_rate: float = 0.1,
        # head
        head_hidden_dim: int = 256,
        head_drop_rate: float = 0.1,
        # optimiser
        lr: float = 5e-4,
        weight_decay: float = 1e-3,
        lr_patience: int = 10,
        lr_factor: float = 0.5,
        lr_min: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()
        if model_name == "UnetHeteroGVPForLBA":
            self.model = UnetHeteroGVPForLBA(
                s_dim=s_dim,
                v_dim=v_dim,
                s_dim_edge=s_dim_edge,
                v_dim_edge=v_dim_edge,
                r_max=r_max,
                num_bessel=num_bessel,
                num_polynomial_cutoff=num_polynomial_cutoff,
                num_layers=num_layers,
                pool=pool,
                fps_ratio=fps_ratio,
                cross_cutoff=cross_cutoff,
                enc_drop_rate=enc_drop_rate,
                head_hidden_dim=head_hidden_dim,
                head_drop_rate=head_drop_rate,
            )
        elif model_name == "UnetHeteroGVPEncoderOnlyForLBA":
            self.model = UnetHeteroGVPEncoderOnlyForLBA(
                s_dim=s_dim,
                v_dim=v_dim,
                s_dim_edge=s_dim_edge,
                v_dim_edge=v_dim_edge,
                r_max=r_max,
                num_bessel=num_bessel,
                num_polynomial_cutoff=num_polynomial_cutoff,
                num_layers=num_layers,
                pool=pool,
                fps_ratio=fps_ratio,
                cross_cutoff=cross_cutoff,
                enc_drop_rate=enc_drop_rate,
                head_hidden_dim=head_hidden_dim,
                head_drop_rate=head_drop_rate,
            )
        
        # Metrics — instantiate one set per phase to avoid state leakage
        for phase in ("train", "val", "test"):
            setattr(self, f"{phase}_rmse", MeanSquaredError(squared=False))
            setattr(self, f"{phase}_mae",  MeanAbsoluteError())
            setattr(self, f"{phase}_pearson", PearsonCorrCoef())

    # ---------------------------------------------------------------- #
    # Shared step                                                        #
    # ---------------------------------------------------------------- #

    def _step(self, batch, phase: str):
        if batch is None:          # entire batch failed featurisation (very rare)
            return None

        out  = self.model(batch)
        loss = out["loss"]
        pred = out["pred"].detach()
        tgt  = batch["graph_y"].float()

        getattr(self, f"{phase}_rmse")(pred, tgt)
        getattr(self, f"{phase}_mae")(pred, tgt)
        getattr(self, f"{phase}_pearson")(pred, tgt)

        self.log(f"{phase}/loss", loss, prog_bar=(phase == "train"),
                 batch_size=tgt.shape[0], on_step=(phase == "train"), on_epoch=True)
        return loss

    def _epoch_end(self, phase: str):
        rmse    = getattr(self, f"{phase}_rmse").compute()
        mae     = getattr(self, f"{phase}_mae").compute()
        pearson = getattr(self, f"{phase}_pearson").compute()

        self.log(f"{phase}/rmse",    rmse,    prog_bar=True)
        self.log(f"{phase}/mae",     mae)
        self.log(f"{phase}/pearson", pearson, prog_bar=(phase == "val"))

        getattr(self, f"{phase}_rmse").reset()
        getattr(self, f"{phase}_mae").reset()
        getattr(self, f"{phase}_pearson").reset()

    # ---------------------------------------------------------------- #
    # Lightning hooks                                                    #
    # ---------------------------------------------------------------- #

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def on_train_epoch_end(self):
        self._epoch_end("train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def on_test_epoch_end(self):
        self._epoch_end("test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.hparams.lr_patience,
            factor=self.hparams.lr_factor,
            min_lr=self.hparams.lr_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/rmse",
                "interval": "epoch",
                "frequency": 1,
            },
        }


# ------------------------------------------------------------------ #
# Main                                                                  #
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser("Train UnetHeteroGVP on LBA")

    # Data
    p.add_argument("--split",       type=int,   default=30, choices=[30, 60])
    p.add_argument("--cutoff",      type=float, default=6.0)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--data_root",   type=str,
                   default="/scratch/ziyiz14/data/GNN_Unet/data")

    # Model — encoder
    p.add_argument("--model_name",        type=str,   default="UnetHeteroGVPForLBA")
    p.add_argument("--s_dim",        type=int,   default=128)
    p.add_argument("--v_dim",        type=int,   default=16)
    p.add_argument("--s_dim_edge",   type=int,   default=32)
    p.add_argument("--v_dim_edge",   type=int,   default=1)
    p.add_argument("--num_layers",   type=int,   default=5)
    p.add_argument("--pool",         type=str,   default="mean", choices=["sum", "mean"])
    p.add_argument("--fps_ratio",    type=float, default=0.6)
    p.add_argument("--enc_drop",     type=float, default=0.1)

    # Model — head
    p.add_argument("--head_hidden",  type=int,   default=256)
    p.add_argument("--head_drop",    type=float, default=0.1)

    # Optimiser
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--wd",           type=float, default=1e-3)
    p.add_argument("--lr_patience",  type=int,   default=10)
    p.add_argument("--lr_factor",    type=float, default=0.5)

    # Training
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--devices",      type=int,   default=1)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--no_test",      action="store_true")
    p.add_argument("--ckpt_path",    type=str,   default=None)

    # Logging
    p.add_argument("--name",         type=str,   default=None)
    p.add_argument("--wandb",        action="store_true")
    p.add_argument("--wandb_project",type=str,   default="lba_unet_hetero_gvp")
    p.add_argument("--log_dir",      type=str,   default="runs/lba")

    return p.parse_args()


def main():
    args = parse_args()
    L.seed_everything(args.seed)

    # ---------------------------------------------------------------- #
    # Run name                                                           #
    # ---------------------------------------------------------------- #
    run_name = args.name or (
        f"split{args.split}_L{args.num_layers}_s{args.s_dim}"
        f"_lr{args.lr}_wd{args.wd}"
    )
    print(f"\n{'='*60}")
    print(f"  Run: {run_name}")
    print(f"{'='*60}\n")

    # ---------------------------------------------------------------- #
    # DataModule                                                         #
    # ---------------------------------------------------------------- #
    dm = LBAHeteroDataModule(
        lba_split=args.split,
        cross_cutoff=args.cutoff,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
    )

    # ---------------------------------------------------------------- #
    # Model                                                              #
    # ---------------------------------------------------------------- #
    model = LBALightningModule(
        model_name=args.model_name,
        s_dim=args.s_dim,
        v_dim=args.v_dim,
        s_dim_edge=args.s_dim_edge,
        v_dim_edge=args.v_dim_edge,
        num_layers=args.num_layers,
        pool=args.pool,
        fps_ratio=args.fps_ratio,
        cross_cutoff=args.cutoff,
        enc_drop_rate=args.enc_drop,
        head_hidden_dim=args.head_hidden,
        head_drop_rate=args.head_drop,
        lr=args.lr,
        weight_decay=args.wd,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
    )

    # ---------------------------------------------------------------- #
    # Loggers                                                            #
    # ---------------------------------------------------------------- #
    loggers = [CSVLogger(save_dir=args.log_dir, name=run_name)]
    if args.wandb:
        loggers.append(
            WandbLogger(
                project=args.wandb_project,
                name=run_name,
                save_dir=args.log_dir,
            )
        )

    # ---------------------------------------------------------------- #
    # Callbacks                                                          #
    # ---------------------------------------------------------------- #
    ckpt_dir = os.path.join(args.log_dir, run_name, "checkpoints")
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-{epoch:03d}-{val/rmse:.4f}",
            monitor="val/rmse",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/rmse",
            patience=25,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ---------------------------------------------------------------- #
    # Trainer                                                            #
    # ---------------------------------------------------------------- #
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        gradient_clip_val=0.5,         # clip gradients: prevents explosion with GVP
        deterministic=False,           # FPS is non-deterministic
    )

    # ---------------------------------------------------------------- #
    # Lazy-layer initialization (run one batch before training)         #
    # ---------------------------------------------------------------- #
    print("Initializing lazy layers ...")
    dm.setup()
    init_batch = next(iter(dm.val_dataloader()))
    with torch.no_grad():
        model.model(init_batch)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}\n")

    # ---------------------------------------------------------------- #
    # Train                                                              #
    # ---------------------------------------------------------------- #
    trainer.fit(model, dm, ckpt_path=args.ckpt_path)

    # ---------------------------------------------------------------- #
    # Test                                                               #
    # ---------------------------------------------------------------- #
    if not args.no_test:
        print("\nRunning test set evaluation ...")
        trainer.test(model, dm, ckpt_path="best")


if __name__ == "__main__":
    main()
