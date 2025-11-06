from torch import nn, optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from utils import *
from model import *

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==== NEW: LightningModule per RICOSTRUZIONE ====
class LitReconSpectral(pl.LightningModule):
    """
    Training end-to-end per ricostruzione spettrale:
    - measurement: due filtri learnable + CSV sensore -> y ∈ (B,8)
    - decoder: MLP 8 -> 121
    Target: s_true ∈ (B,121)
    """
    def __init__(self, spectral_sens_csv: str, lr: float = 1e-3, out_len: int = 121):
        super().__init__()
        self.save_hyperparameters()
        self.meas = DualFilterVector(spectral_sens_csv, device="cuda", dtype=torch.float32)
        self.dec = ReconMLP(in_dim=8, out_len=out_len)
        self.lr = lr

    def forward(self, s_true):
        y = self.meas(s_true)     # (B,8)
        s_pred = self.dec(y)      # (B,121)
        return s_pred, y

    def _metrics(self, s_pred, s_true):
        with torch.no_grad():
            mse  = F.mse_loss(s_pred, s_true)
            rmse = torch.sqrt(mse)
            sam_rad = spectral_angle_loss(s_pred, s_true)
            sam_deg = sam_rad * (180.0 / math.pi)
        return mse, rmse, sam_deg

    def _loss(self, s_pred, s_true):
        loss = F.mse_loss(s_pred, s_true)
        loss += 0.1 * spectral_angle_loss(s_pred, s_true)
        loss += 0.01 * spectral_smoothness(s_pred)
        loss += 0.01 * self.meas.filters_smoothness()
        return loss

    def training_step(self, batch, batch_idx):
        # batch: (x, _) ma qui usiamo x come spettro vero (B,121)
        x, _ = batch
        # Se il dataset ti fornisce (B, C, H, W), media su H,W:
        if x.dim() == 4:
            x = x.mean(dim=(2,3))
        s_true = x
        s_pred, _ = self(s_true)
        loss = self._loss(s_pred, s_true)
        mse, rmse, sam_deg = self._metrics(s_pred, s_true)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_rmse", rmse, prog_bar=False)
        self.log("train_sam_deg", sam_deg, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        if x.dim() == 4:
            x = x.mean(dim=(2,3))
        s_true = x
        s_pred, _ = self(s_true)
        loss = self._loss(s_pred, s_true)
        mse, rmse, sam_deg = self._metrics(s_pred, s_true)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_rmse", rmse, prog_bar=True)
        self.log("val_sam_deg", sam_deg, prog_bar=True)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-6)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=50)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

# ---------------- main ----------------
def main(
        data_root: str,
        sensor_root: str,                # NEW: "recon" o "classif"
        save_dir: str = "runs/lightning",
        batch_size: int = 8,
        num_workers: int = 4,
        lr: float = 1e-3,
        epochs: int = 50,
        seed: int = 42,
        devices="auto"):
    set_seed(seed)

    # Per ricostruzione: vogliamo (B,121). Se il dataset emette patch (B,C,H,W),
    # passiamo patch_mean=True così x -> media su H,W in __getitem__ (se già supportato).
    train_loader, val_loader = make_loaders(data_root, batch_size=batch_size, num_workers=num_workers, val_ratio=0.2)

    model = LitReconSpectral(spectral_sens_csv=sensor_root, lr=lr, out_len=121)
    run_dir = f"{save_dir}/recon"
    monitor_key = "val_loss"

    ckpt = ModelCheckpoint(dirpath=run_dir, filename="best", monitor=monitor_key, mode="min", save_top_k=1)
    early = EarlyStopping(monitor=monitor_key, mode="min", patience=100, min_delta=0.0, verbose=True)
    lrmon = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        default_root_dir=run_dir,
        max_epochs=epochs,
        accelerator="auto",
        devices=devices,
        precision="32",
        callbacks=[ckpt, early, lrmon],
        log_every_n_steps=10,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--sensor_root", type=str)   # richiesto in recon
    ap.add_argument("--save_dir", type=str, default="runs/lightning")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    main(
        data_root=args.data_root,
        sensor_root=args.sensor_root,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed
    )
