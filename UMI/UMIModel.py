###############################################################################
# High-level UMI wrapper – engineered for live, back-test and HPC use         #
###############################################################################
#
# NOTE
# ----
# • Relies on the low-level blocks already defined earlier in main.py
#   (StockLevelFactorLearning, MarketLevelFactorLearning, UMIForecastingModel)
# • Uses PyTorch DistributedDataParallel (DDP) if torch.distributed has been
#   initialised by `torchrun` / `mpirun … python -m torch.distributed.run …`.
# • Optional Optuna hyper-parameter tuning when `tune_hparams=True`.
#
###############################################################################

import os, math, json, shutil, datetime as dt
from pathlib import Path
from typing import Dict, Optional, Any

from modules import StockLevelFactorLearning, MarketLevelFactorLearning, UMIForecastingModel
import pandas as pd
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import optuna
from sqlalchemy.exc import OperationalError
from optuna.storages import RDBStorage
# --------------------------------------------------------------------------- #
# small helper – convert a frequency string like “15m” → pandas offset alias #
# --------------------------------------------------------------------------- #
_FREQ2PANDAS = {
    "1m": "T", "5m": "5T", "15m": "15T", "30m": "30T",
    "1h": "H", "2h": "2H", "4h": "4H",
    "1d": "D",
}

DEFAULT_DB_URL = "sqlite:///umi_hp_optimization.db"

# --------------------------------------------------------------------------- #
# 1. Dataset that yields sliding windows ready for the factor blocks          #
# --------------------------------------------------------------------------- #
class SlidingWindowDataset(Dataset):
    """
    Yields tuples:
      prices_seq : (L+1, I)
      feat_seq   : (L+1, I, F)
      target     : (I, 1)   -- next-bar return (t = L)  [optional]
    """
    def __init__(
        self,
        panel: torch.Tensor,      # (T, I, F) F includes "close" as first idx
        window_len: int,
        pred_len: int,
        close_idx: int,
        with_target: bool = True,
    ):
        self.panel = panel
        self.L = window_len
        self.pred = pred_len
        self.close_idx = close_idx
        self.with_target = with_target

    def __len__(self):
        return self.panel.size(0) - self.L - self.pred + 1

    def __getitem__(self, idx):
        seq = self.panel[idx : idx + self.L + 1]             # (L+1,I,F)
        prices = seq[..., self.close_idx]                    # (L+1,I)
        if self.with_target:
            tgt_close = self.panel[idx + self.L : idx + self.L + self.pred,
                                    :, self.close_idx]       # (pred,I)
            ret = (tgt_close[-1] - tgt_close[0]) / (tgt_close[0] + 1e-8)
            ret = ret.unsqueeze(-1)                          # (I,1)
            return prices, seq, ret
        return prices, seq

# --------------------------------------------------------------------------- #
# 2. Utility : build a panel tensor from the {stockID: dataframe} dict        #
# --------------------------------------------------------------------------- #
def build_panel(data_dict: Dict[str, "pd.DataFrame"]) -> torch.Tensor:
    """
    Returns tensor of shape (T, I, F) sorted by timestamp ascending &
    stocks alphabetically.
    Returns
    tensor : (T, I, F)
    idx    : DatetimeIndex (UTC)
    """

    # 1) align all dataframes on inner join of timestamps
    keys = sorted(data_dict)
    # --- harmonise indices -------------------------------------------------
    for k, df in data_dict.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            df.set_index("Date", inplace=True)
        df.index = pd.to_datetime(df.index, utc=True)  # FIX – same clock :contentReference[oaicite:1]{index=1}

    feature_cols = data_dict[keys[0]].columns          # all numeric + tech-indicators
    long = pd.concat([df[feature_cols].assign(stock=k) for k, df in data_dict.items()])

    # ---------- proper pivot ---------------------------------------------- #
    df_pivot = long.pivot_table(
        values=feature_cols,      # every feature
        index=long.index,         # <-- FIX 3: pivot on real index :contentReference[oaicite:2]{index=2}
        columns="stock"
    ).sort_index()

    tensor = torch.tensor(df_pivot.values, dtype=torch.float32)
    T, F, I = len(df_pivot), len(feature_cols), len(keys)
    tensor = tensor.reshape(T, F, I).transpose(0, 2, 1)  # → (T, I, F)

    return tensor, df_pivot.index       

# --------------------------------------------------------------------------- #
# 3. Main model                                                               #
# --------------------------------------------------------------------------- #
class UMIModel:
    """
    Orchestrates factor learning + forecasting, periodic retraining, optional
    hyper-parameter tuning, DDP-aware saving/loading.  Designed for both
    back-tests and always-on live services.
    """

    # ---------------- constructor ------------------------------------ #
    def __init__(
        self,
        freq: str,
        feature_dim: int,
        window_len: int,
        pred_len: int,
        end_train: str,
        end_valid: str,
        end_test: Optional[str],
        retrain_every: str = "30d",
        tune_hparams: bool = False,
        tune_trials: int = 50,
        self.n_epochs: int = 20,
        **hparams,
    ):
        self.freq           = freq
        self.F              = feature_dim
        self.L              = window_len
        self.pred_len       = pred_len
        self.retrain_delta  = pd.Timedelta(retrain_every)
        self.end_train      = pd.Timestamp(end_train)
        self.end_valid      = pd.Timestamp(end_valid)
        self.end_test       = pd.Timestamp(end_test) if end_test else None
        self.tune           = tune_hparams
        self.trials         = tune_trials
        self.hp             = self._default_hparams()
        self.hp.update(hparams)
        self._tuning_done = False   # flag to avoid re-tuning of hyper-params

        self._device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self._world_size    = dist.get_world_size() if dist.is_initialized() else 1
        #self._rank          = dist.get_rank() if dist.is_initialized() else 0

        self._last_fit_time = None
        self._model_dir     = None

    # ---------------- hyper-param defaults --------------------------- #
    def _default_hparams(self) -> Dict[str, Any]:
        return dict(
            lambda_ic     = 0.1,
            lambda_sync   = 1.0,
            lambda_rankic = 0.1,
            temperature   = 0.07,
            sync_thr      = 0.6,
            lr            = 1e-3,
            weight_decay  = 0.0,
        )

    # ---------------- sub-module builder ----------------------------- #
    def _build_submodules(self):
        I = self.I if hasattr(self, 'I') else 0  # I is set in fit()

        self.stock_factor = StockLevelFactorLearning(
            I, lambda_ic=self.hp["lambda_ic"]
        ).to(self._device)

        self.market_factor = MarketLevelFactorLearning(
            I, self.F, window_L=self.L,
            lambda_sync=self.hp["lambda_sync"],
            temperature=self.hp["temperature"],
            sync_threshold=self.hp["sync_thr"],
        ).to(self._device)

        self.forecaster = UMIForecastingModel(
            I, self.F, u_dim=1,
            W_iota=self.market_factor.W_iota,
            pred_len=self.pred_len,
            lambda_rankic=self.hp["lambda_rankic"],
        ).to(self._device)

        # Wrap with DDP if needed
        #if dist.is_initialized():
        #    self.stock_factor  = nn.parallel.DistributedDataParallel(
        #        self.stock_factor, device_ids=[self._rank]
        #    )
        #    self.market_factor = nn.parallel.DistributedDataParallel(
        #        self.market_factor, device_ids=[self._rank]
        #    )
        #    self.forecaster    = nn.parallel.DistributedDataParallel(
        #        self.forecaster, device_ids=[self._rank]
        #    )

    # ---------------- training utilities ----------------------------- #
    def _train_epoch(self, loader, optim_s, optim_m, optim_f):
        self.stock_factor.train(); self.market_factor.train(); self.forecaster.train()
        total = 0
        for prices_seq, feat_seq, target in loader:
            prices_seq = prices_seq.to(self._device)
            feat_seq   = feat_seq.to(self._device)
            target     = target.to(self._device)

            # Stage-1
            u_seq, r_t, m_t, loss_factors, _ = self._stage1_forward(prices_seq, feat_seq)
            optim_s.zero_grad(set_to_none=True)
            optim_m.zero_grad(set_to_none=True)
            loss_factors.backward()
            optim_s.step(); optim_m.step()

            # Stage-2
            preds, loss_pred, _ = self._stage2_forward(feat_seq, u_seq, r_t, m_t, target)
            optim_f.zero_grad(set_to_none=True)
            loss_pred.backward()
            optim_f.step()

            total += loss_pred.item()
        return total / len(loader)

    # ---------------- stage1 / stage2 wrappers ----------------------- #
    def _stage1_forward(self, prices_seq, feat_seq):
        stockIDs = torch.eye(prices_seq.size(2), device=prices_seq.device)
        return self.market_factor.stage1_forward(prices_seq, feat_seq, stockIDs)

    def _stage2_forward(self, feat_seq, u_seq, r_t, m_t, target):
        stockIDs = torch.eye(feat_seq.size(2), device=feat_seq.device)
        return self.forecaster(
            feat_seq, u_seq, r_t, m_t, stockIDs, target
        )

    # ---------------- fit -------------------------------------------- #
    def fit(self, data_dict: Dict[str, "pd.DataFrame"]):
        """
        One-off training (train + valid).  Optionally tunes hyper-parameters
        on the validation set using Optuna.
        """

        # build loaders
        panel, idx = build_panel(data_dict)   # panel.shape: (T,I,F)
        self.I = panel.size(1)
        self._build_submodules() # build sub-modules with I

        train_mask = idx <= self.end_train
        valid_mask = (idx > self.end_train) & (idx <= self.end_valid)

        ds_train = SlidingWindowDataset(panel[train_mask], self.L, self.pred_len, close_idx=0)
        ds_valid = SlidingWindowDataset(panel[valid_mask], self.L, self.pred_len, close_idx=0)

        train_sampler = (
            torch.utils.data.distributed.DistributedSampler(ds_train)
            if dist.is_initialized() else None
        )
        loader_train = DataLoader(
            ds_train, batch_size=32, sampler=train_sampler, shuffle=train_sampler is None
        )
        loader_valid = DataLoader(ds_valid, batch_size=32)

        # optuna objective ----------------------------------------------------
        def objective(trial):
            #if dist.is_initialized() and dist.get_rank() != 0:
            #    return float("inf")  # only rank-0 does tuning

            # sample hyper-params
            for key, rng in [
                ("lambda_ic",   (0.01, 1.0)),
                ("lambda_sync", (0.1,  5.0)),
                ("lambda_rankic", (0.01, 1.0)),
                ("temperature", (0.03, 0.2)),
                ("sync_thr",    (0.5, 0.8)),
                ("lr",          (1e-4, 5e-3)),
            ]:
                self.hp[key] = trial.suggest_float(key, *rng, log=True) if key == "lr" \
                               else trial.suggest_float(key, *rng)

            self._build_submodules()
            opt_s = torch.optim.AdamW(self.stock_factor.parameters(),  lr=self.hp["lr"])
            opt_m = torch.optim.AdamW(self.market_factor.parameters(), lr=self.hp["lr"])
            opt_f = torch.optim.AdamW(self.forecaster.parameters(),    lr=self.hp["lr"])

            # single epoch for speed; you can loop more
            self._train_epoch(loader_train, opt_s, opt_m, opt_f)

            # validation
            self.stock_factor.eval(); self.market_factor.eval(); self.forecaster.eval()
            mse_valid = 0
            with torch.no_grad():
                for p_seq, f_seq, tgt in loader_valid:
                    p_seq, f_seq, tgt = p_seq.to(self._device), f_seq.to(self._device), tgt.to(self._device)
                    u_seq, r_t, m_t, _, _ = self._stage1_forward(p_seq, f_seq)
                    preds, loss, _ = self._stage2_forward(f_seq, u_seq, r_t, m_t, tgt)
                    mse_valid += loss.item()
            return mse_valid / len(loader_valid)

        # run tuning or just plain training
        if self.tune and not self._tuning_done:
            db_url     = os.getenv("STORAGE_URL", DEFAULT_DB_URL)
            study_name = os.getenv("STUDY_NAME", "umi_opt")
            storage = RDBStorage(              
                db_url,
                engine_kwargs={"connect_args": {"timeout": 30}, "pool_pre_ping": True}
                )


            study = optuna.create_study(study_name=study_name, storage=storage, direction="minimize", load_if_exists=True)
            study.optimize(objective, n_trials=self.trials)
            self.hp.update(study.best_params)
            #if self._rank == 0:
            print("Best hyper-params:", study.best_params)

        # final train on train+valid
        self._build_submodules()
        opt_s = torch.optim.AdamW(self.stock_factor.parameters(),  lr=self.hp["lr"],
                                  weight_decay=self.hp["weight_decay"])
        opt_m = torch.optim.AdamW(self.market_factor.parameters(), lr=self.hp["lr"],
                                  weight_decay=self.hp["weight_decay"])
        opt_f = torch.optim.AdamW(self.forecaster.parameters(),    lr=self.hp["lr"],
                                  weight_decay=self.hp["weight_decay"])

        # simple training loop
        for epoch in range(self.n_epochs):  # small epoch count for brevity
            loss_epoch = self._train_epoch(loader_train, opt_s, opt_m, opt_f)
            #if self._rank == 0:
            print(f"epoch {epoch} train_loss {loss_epoch:.4f}")

        # save
        #if self._rank == 0:
        timestamp = dt.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
        self._model_dir = Path(f"models/{self.freq}/{timestamp}")
        self._model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), self._model_dir / "best.pt")
        with open(self._model_dir / "hparams.json", "w") as fp:
            json.dump(self.hp, fp, indent=2)
        self._last_fit_time = pd.Timestamp.utcnow()

    # ------------------------------------------------------------------ #
    # update : decide if retrain_every elapsed → refit on latest data    #
    # ------------------------------------------------------------------ #
    def update(self, new_data_dict: Dict[str, "pd.DataFrame"]):
        if self._last_fit_time is None:
            raise RuntimeError("Call fit() before update().")

        now = pd.Timestamp.utcnow()     #### change to current time backtesting or live
        if now - self._last_fit_time >= self.retrain_delta:
            print("Retraining triggered …")
            self.fit(new_data_dict)

    # ------------------------------------------------------------------ #
    # predict : one inference step                                       #
    # ------------------------------------------------------------------ #
    def predict(self, latest_dict: Dict[str, "pd.DataFrame"]) -> Dict[str, float]:
        panel = build_panel(latest_dict)                     # (T,I,F)
        assert panel.size(0) >= self.L + 1, "need L+1 bars for inference"

        prices_seq = panel[-self.L-1:, :, 0]                 # (L+1,I)
        feat_seq   = panel[-self.L-1:]                       # (L+1,I,F)
        prices_seq = prices_seq.to(self._device)
        feat_seq   = feat_seq.to(self._device)

        with torch.no_grad():
            u_seq, r_t, m_t, _, _ = self._stage1_forward(prices_seq.unsqueeze(0),
                                                         feat_seq.unsqueeze(0))
            preds, _, _ = self._stage2_forward(
                feat_seq.unsqueeze(0), u_seq, r_t, m_t,
                target=None
            )
        # return as dict {stock: prediction}
        keys = sorted(latest_dict.keys())
        return {k: float(preds[0, i].cpu()) for i, k in enumerate(keys)}

    # ------------------------------------------------------------------ #
    # state-dict helpers (single-process save/load)                       #
    # ------------------------------------------------------------------ #
    def state_dict(self) -> Dict[str, Any]:
        return {
            "stock_factor":  self.stock_factor.module.state_dict()
            if isinstance(self.stock_factor, nn.parallel.DistributedDataParallel)
            else self.stock_factor.state_dict(),
            "market_factor": self.market_factor.module.state_dict()
            if isinstance(self.market_factor, nn.parallel.DistributedDataParallel)
            else self.market_factor.state_dict(),
            "forecaster":    self.forecaster.module.state_dict()
            if isinstance(self.forecaster, nn.parallel.DistributedDataParallel)
            else self.forecaster.state_dict(),
            "hparams": self.hp,
        }

    def load_state_dict(self, sd: Dict[str, Any]):
        self.hp.update(sd["hparams"])
        self._build_submodules()
        self.stock_factor.load_state_dict(sd["stock_factor"])
        self.market_factor.load_state_dict(sd["market_factor"])
        self.forecaster.load_state_dict(sd["forecaster"])

# ------------------------------------------------------------------ #
#  Main guard: lets the file act as CLI for SLURM array jobs         #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import argparse, json, pandas as pd, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-json",   default="data/ohlcv_plus_indicators.json",
                    help="dict-of-dataframes serialized via DataFrame.to_json")
    ap.add_argument("--freq",        default="1d")
    ap.add_argument("--window-len",  type=int, default=60)
    ap.add_argument("--pred-len",    type=int, default=1)
    ap.add_argument("--end-train",   default="2019-12-31")
    ap.add_argument("--end-valid",   default="2020-12-31")
    ap.add_argument("--study-name",  default=os.getenv("STUDY_NAME", "umi"))
    ap.add_argument("--storage-url", default=os.getenv("STORAGE_URL", "sqlite:///umi_hp_optim.db"))
    ap.add_argument("--n-epochs",    type=int, default=20)
    ap.add_argument("--tune-hparams", type=bool, default=False)
    ap.add_argument("--tune-trials", type=int, default=50)
    ap.add_argument("--retrain-every", default="30d")
    ap.add_argument("--end-test",    default=None, help="optional end date for test set")
    
    args = ap.parse_args()

    # ---------- load data -------------------------------------------
    with open(args.data_json) as fp:
        raw = json.load(fp)
    data_dict = {k: pd.read_json(v, convert_dates=True) for k, v in raw.items()}

    # ---------- build & fit -----------------------------------------
    # Each SLURM array task sees exactly one GPU thanks to CUDA_VISIBLE_DEVICES
    model = UMIModel(
        freq=args.freq,
        feature_dim=len(data_dict[next(iter(data_dict))].columns),
        window_len=args.window_len,
        pred_len=args.pred_len,
        end_train=args.end_train,
        end_valid=args.end_valid,
        end_test=None,
        tune_hparams=True,          # first run tunes
        tune_trials=0,              # number controlled per-job via TRIALS_PER_JOB env
        study_name=args.study_name,
        storage_url=args.storage_url,
    )

    model.fit(data_dict)            # Optuna tuning occurs only once per study
