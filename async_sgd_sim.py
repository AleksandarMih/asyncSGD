"""
Async SGD — gradient-based parameter server.

Flow per worker per batch:
  1. Pull latest weights from PS
  2. Forward + backward  →  flat gradient vector
  3. Tag gradient with worker's current step number
  4. Compress with RHT (optional, --compress-bits 2|4|8)
  5. Send to PS with simulated delay

PS per received gradient:
  1. Check staleness: worker_step must be within (ps_step - 3*N, ps_step]
     If too stale: reject, re-queue batch, return current weights
  2. Apply gradient via optimizer.step()
  3. Record weight delta → append to suffix-sum history
  4. Return to worker: suffix sum of all deltas since worker's step
     so worker can fast-forward its local copy without full weight sync

Suffix sum:
  history = [(step_i, delta_i), ...]  bounded deque of length 3*N
  sfx[i]  = delta_i + delta_{i+1} + ... + delta_latest
  Worker at step k gets sfx[k+1], adds it to local weights.
  If worker is too far behind the window: send full weights instead.

Compression (RHT, THC-style):
  Worker side:  g_send = compress(g + error_buf);  error_buf = g - decompress(g_send)
  PS side:      g_recv = decompress(g_send)
  No norm exchange, no homomorphic sum — we decompress on PS (simpler, correct).
  THC's homomorphic trick only matters when PS is a switch with no float ops.

WandB:
  worker_{i}/train_loss    per batch
  worker_{i}/grad_norm     ||g||_2
  worker_{i}/rejections    cumulative
  merger/eval_loss         per PS step
  merger/eval_acc          per PS step

Usage:
  pip install torch torchvision transformers wandb
  python async_sgd_sim.py
  python async_sgd_sim.py --num_workers 4 --compress-bits 4
  python async_sgd_sim.py --no-compress
  python async_sgd_sim.py --num_workers 3 --delay_dist poisson --delay_param 0.5
"""

import asyncio
import argparse
import collections
import copy
import math
import time
import random
import threading
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from transformers import AutoModelForImageClassification
import wandb


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    num_workers:    int   = 2
    delay_dist:     str   = "exp"
    delay_param:    float = 0.3
    epochs:         int   = 3
    batch_size:     int   = 64
    lr:             float = 1e-3
    momentum:       float = 0.9
    weight_decay:   float = 1e-4
    seed:           int   = 42
    data_root:      str   = "./data"
    wandb_project:  str   = "async-sgd-sim"
    wandb_run_name: str | None = None
    compress:       bool  = True
    compress_bits:  int   = 8       # 2=16x  4=8x  8=4x compression


# ---------------------------------------------------------------------------
# RHT Compression  (invisible to the rest of the code)
# ---------------------------------------------------------------------------

class RHTCodec:
    """
    Randomized Hadamard Transform + stochastic b-bit uniform quantization.
    Used only as a compression layer — caller just calls encode/decode.

    Error feedback is managed by the worker (see WorkerState).
    """

    def __init__(self, bits: int = 8, seed: int = 42):
        self.bits   = bits
        self.levels = 2 ** bits - 1
        self._signs: dict[int, torch.Tensor] = {}
        self._rng   = torch.Generator()
        self._rng.manual_seed(seed)

    @staticmethod
    def _next_pow2(n: int) -> int:
        return 1 if n == 0 else 2 ** math.ceil(math.log2(max(n, 1)))

    @staticmethod
    def _wht(x: torch.Tensor) -> torch.Tensor:
        """Vectorized Walsh-Hadamard Transform. Length must be power of 2."""
        n = x.shape[0]
        assert n & (n - 1) == 0
        while n > 1:
            x = x.reshape(-1, 2)
            a, b = x[:, 0], x[:, 1]
            x = torch.stack([a + b, a - b], dim=1).reshape(-1)
            n //= 2
        return x / math.sqrt(x.shape[0])

    def _get_signs(self, n: int) -> torch.Tensor:
        if n not in self._signs:
            s = (torch.randint(0, 2, (n,), generator=self._rng) * 2 - 1).float()
            self._signs[n] = s
        return self._signs[n]

    def encode(self, x: torch.Tensor) -> dict:
        """Compress a flat float32 gradient vector."""
        orig_len = x.shape[0]
        pad_len  = self._next_pow2(orig_len)

        y = torch.zeros(pad_len)
        y[:orig_len] = x.float()
        y = y * self._get_signs(pad_len)
        y = self._wht(y)

        # Stochastic uniform quantization in [-norm, +norm]
        norm = y.abs().max().item()
        if norm < 1e-12:
            return {"data": torch.zeros(orig_len, dtype=torch.uint8),
                    "norm": 1.0, "orig_len": orig_len}

        v     = (y / norm + 1.0) * (self.levels / 2.0)       # [0, levels]
        floor = v.floor()
        frac  = v - floor
        q     = (floor + (torch.rand_like(frac) < frac).float())
        q     = q.clamp(0, self.levels).to(torch.uint8)

        return {"data": q, "norm": norm, "orig_len": orig_len}

    def decode(self, pkg: dict) -> torch.Tensor:
        """Decompress back to flat float32."""
        q        = pkg["data"].float()
        norm     = pkg["norm"]
        orig_len = pkg["orig_len"]
        pad_len  = q.shape[0]

        y = (q / (self.levels / 2.0) - 1.0) * norm

        # Inverse WHT: WHT is self-inverse up to scaling
        n = pad_len
        while n > 1:
            y = y.reshape(-1, 2)
            a, b = y[:, 0], y[:, 1]
            y = torch.stack([a + b, a - b], dim=1).reshape(-1)
            n //= 2
        y = y / math.sqrt(pad_len) * math.sqrt(pad_len)  # cancel WHT norm
        y = y / math.sqrt(pad_len)                        # reapply for inverse
        y = y * self._get_signs(pad_len)

        return y[:orig_len]

    @property
    def compression_ratio(self) -> float:
        return 32.0 / self.bits


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def build_model(num_labels: int = 10) -> nn.Module:
    model = AutoModelForImageClassification.from_pretrained(
        "microsoft/resnet-18", num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    for m in model.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    return model


def flat_grads(model: nn.Module) -> torch.Tensor:
    """Concatenate all parameter gradients into one flat CPU vector."""
    return torch.cat([
        p.grad.float().flatten().cpu()
        for p in model.parameters()
        if p.grad is not None
    ])


def flat_params(model: nn.Module) -> torch.Tensor:
    """Concatenate all parameters into one flat CPU vector."""
    return torch.cat([p.data.float().flatten().cpu()
                      for p in model.parameters()])


def load_flat_params(model: nn.Module, flat: torch.Tensor,
                     device: torch.device):
    """Write a flat vector back into model parameters."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset:offset + n].reshape(p.shape).to(device))
        offset += n


# ---------------------------------------------------------------------------
# Batch Distributor
# ---------------------------------------------------------------------------

class BatchDistributor:
    def __init__(self, n_samples: int, batch_size: int,
                 epochs: int, seed: int):
        self.n_samples  = n_samples
        self.batch_size = batch_size
        self.epochs     = epochs
        self._rng       = np.random.default_rng(seed)
        self._queue: collections.deque[list[int]] = collections.deque()
        self._epoch = 0
        self._fill()

    def _fill(self):
        idx = self._rng.permutation(self.n_samples).tolist()
        for i in range(0, len(idx), self.batch_size):
            chunk = idx[i:i + self.batch_size]
            if chunk:
                self._queue.append(chunk)
        self._epoch += 1
        print(f"  [dist] epoch {self._epoch}: {len(self._queue)} batches")

    def next_batch(self) -> list[int] | None:
        if not self._queue:
            if self._epoch >= self.epochs:
                return None
            self._fill()
        return self._queue.popleft() if self._queue else None

    def return_batch(self, indices: list[int]):
        """Re-queue a rejected batch at the front."""
        self._queue.appendleft(indices)


# ---------------------------------------------------------------------------
# Eval thread
# ---------------------------------------------------------------------------

def run_eval(model, val_loader, device, step):
    try:
        model = model.to(device)
        model.eval()
        crit = nn.CrossEntropyLoss()
        loss_sum = correct = total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out      = model(imgs).logits
                loss_sum += crit(out, lbls).item() * lbls.size(0)
                correct  += (out.argmax(1) == lbls).sum().item()
                total    += lbls.size(0)
        wandb.log({"merger/eval_loss": loss_sum / total,
                   "merger/eval_acc":  correct  / total}, step=step)
        print(f"  [eval] step={step} loss={loss_sum/total:.4f} "
              f"acc={correct/total:.4f}")
    finally:
        model.cpu()
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Parameter Server
# ---------------------------------------------------------------------------

class ParameterServer:
    """
    Holds model + optimizer. Workers push (step_tag, gradient) pairs.

    Suffix-sum history:
      After each optimizer step, record weight delta:
        delta_t = theta_t - theta_{t-1}
      Keep a bounded deque of (ps_step, delta) pairs, length = 3*N.
      Build suffix sums:
        sfx[i] = delta_i + delta_{i+1} + ... + delta_latest
      Worker that last saw step k gets sfx[k+1] added to its weights.
      This is exact catch-up: no full weight transfer needed unless the
      worker is behind the window (extreme laggard).

    Staleness cap: reject if worker_step < ps_step - 3*N.
    """

    def __init__(
        self,
        model:        nn.Module,
        optimizer:    optim.Optimizer,
        val_loader:   DataLoader,
        eval_device:  torch.device,
        distributor:  BatchDistributor,
        num_workers:  int,
        codec:        RHTCodec | None,
    ):
        self.model       = model
        self.optimizer   = optimizer
        self.val_loader  = val_loader
        self.eval_device = eval_device
        self.distributor = distributor
        self.num_workers = num_workers
        self.codec       = codec
        self.max_lag     = 3 * num_workers   # staleness cap

        self._lock = asyncio.Lock()

        # Step counter — incremented after each optimizer.step()
        self.ps_step = 0

        # Per-worker last accepted step
        self._worker_step: dict[int, int] = {}

        # Rejection counters
        self._rejections: dict[int, int] = {}

        # Suffix-sum history: deque of (ps_step_after_update, delta_flat)
        self._history: collections.deque[tuple[int, torch.Tensor]] = \
            collections.deque(maxlen=self.max_lag)
        # Suffix sum cache: sfx_cache[step] = sum of deltas from that step onward
        self._sfx_cache: dict[int, torch.Tensor] = {}

        # Total params count (for flat vector ops)
        self._n_params = sum(p.numel() for p in model.parameters())

    def _rebuild_sfx(self):
        """Recompute suffix sums over the current history window."""
        if not self._history:
            self._sfx_cache = {}
            return
        sfx   = {}
        accum = None
        for step, delta in reversed(self._history):
            accum = delta.clone() if accum is None else accum + delta
            sfx[step] = accum.clone()
        self._sfx_cache = sfx

    def _catch_up(self, worker_step: int) -> torch.Tensor | None:
        """
        Return the flat delta the worker needs to add to its weights
        to reach the current PS state, or None if already up to date.
        Falls back to None (full sync) if worker is behind the window.
        """
        if self.ps_step == worker_step:
            return None  # already current
        if not self._history:
            return None

        oldest_step = self._history[0][0]
        # Worker needs deltas from (worker_step+1) onward
        target = worker_step + 1

        if target < oldest_step:
            # Too far behind the window — caller will send full weights
            return None

        return self._sfx_cache.get(target)

    async def push_grad(
        self,
        worker_id:    int,
        worker_step:  int,       # PS step the worker was at when it computed grad
        grad_payload: dict,      # {"data": ..., "norm": ..., "orig_len": ...}
                                 # or {"flat": tensor} if uncompressed
        batch_indices: list[int],
        delay:        float,
    ) -> dict:
        """
        Receive a gradient from worker_id, apply it, return catch-up payload.

        Returns:
          {"mode": "delta", "delta": tensor}   — worker adds this to its weights
          {"mode": "full",  "flat":  tensor}   — worker replaces its weights
          {"mode": "reject"}                   — rejected (shouldn't normally reach worker)
        """
        await asyncio.sleep(delay)

        async with self._lock:
            # ---- Init worker tracking on first contact ----
            if worker_id not in self._worker_step:
                self._worker_step[worker_id] = 0
                self._rejections[worker_id]  = 0

            # ---- Staleness check ----
            lag = self.ps_step - worker_step
            if lag > self.max_lag and self.ps_step > 0:
                self._rejections[worker_id] += 1
                wandb.log(
                    {f"worker_{worker_id}/rejections":
                     self._rejections[worker_id]},
                    step=self.ps_step,
                )
                self.distributor.return_batch(batch_indices)
                print(f"  [PS] REJECT W{worker_id} "
                      f"(lag={lag} > cap={self.max_lag})")
                # Return full weights so worker can re-sync
                return {"mode": "full",
                        "flat": flat_params(self.model)}

            # ---- Decompress gradient ----
            if self.codec is not None and "flat" not in grad_payload:
                g = self.codec.decode(grad_payload)
            else:
                g = grad_payload["flat"]

            # ---- Apply gradient via optimizer ----
            theta_before = flat_params(self.model).clone()

            self.optimizer.zero_grad()
            offset = 0
            for p in self.model.parameters():
                n = p.numel()
                chunk = g[offset:offset + n].reshape(p.shape)
                p.grad = chunk.to(p.device)
                offset += n
            self.optimizer.step()

            theta_after = flat_params(self.model)
            delta       = theta_after - theta_before

            # ---- Update suffix-sum history ----
            self.ps_step += 1
            self._history.append((self.ps_step, delta.clone()))
            self._rebuild_sfx()
            self._worker_step[worker_id] = self.ps_step

            print(f"  [PS] step={self.ps_step} "
                  f"W{worker_id} lag={lag} "
                  f"|g|={g.norm().item():.4f}")

            # ---- Spawn eval ----
            eval_model = build_model(num_labels=10)
            eval_model.load_state_dict(
                copy.deepcopy(self.model.state_dict())
            )
            threading.Thread(
                target=run_eval,
                args=(eval_model, self.val_loader,
                      self.eval_device, self.ps_step),
                daemon=True,
            ).start()

            # ---- Build catch-up response ----
            catch_up = self._catch_up(worker_step)
            if catch_up is not None:
                return {"mode": "delta", "delta": catch_up.clone()}
            else:
                return {"mode": "full",
                        "flat": flat_params(self.model).clone()}


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class WorkerState:
    def __init__(self):
        self.local_step:  int = 0              # which PS step worker last synced
        self.error_buf:   torch.Tensor | None = None   # RHT error feedback
        self.batch_count: int = 0


async def worker_loop(
    worker_id:   int,
    model:       nn.Module,
    device:      torch.device,
    delay_dist:  str,
    delay_param: float,
    ps:          ParameterServer,
    distributor: BatchDistributor,
    train_ds,
    codec:       RHTCodec | None,
) -> None:
    model = model.to(device)
    crit  = nn.CrossEntropyLoss()
    state = WorkerState()

    worker_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    while True:
        # ---- Get batch indices from distributor (under PS lock) ----
        async with ps._lock:
            indices = distributor.next_batch()
        if indices is None:
            break

        # ---- Load batch ----
        subset = Subset(train_ds, indices)
        loader = DataLoader(subset, batch_size=len(indices),
                            shuffle=False, num_workers=0)
        raw_imgs, labels = next(iter(loader))
        images = torch.stack([worker_tf(img) for img in raw_imgs]).to(device)
        labels = labels.to(device)

        # ---- Forward + backward — NO optimizer.step() ----
        for p in model.parameters():
            p.grad = None
        out  = model(images).logits
        loss = crit(out, labels)
        loss.backward()

        g = flat_grads(model)   # flat CPU float32 gradient
        grad_norm = g.norm().item()
        state.batch_count += 1

        wandb.log({
            f"worker_{worker_id}/train_loss": loss.item(),
            f"worker_{worker_id}/grad_norm":  grad_norm,
        }, step=ps.ps_step)

        # ---- Compress gradient (with error feedback) ----
        if codec is not None:
            # Add error feedback residual from last round
            g_send = g if state.error_buf is None else g + state.error_buf
            pkg    = codec.encode(g_send)
            # Update error buffer: what the quantizer dropped
            g_reconstructed  = codec.decode(pkg)
            state.error_buf  = g_send - g_reconstructed
            payload          = pkg
        else:
            payload         = {"flat": g}
            state.error_buf = None

        delay = sample_delay(delay_dist, delay_param)
        print(f"  [W{worker_id}] batch={state.batch_count} "
              f"step={state.local_step} "
              f"loss={loss.item():.4f} |g|={grad_norm:.3f} "
              f"delay={delay:.3f}s")

        # ---- Push gradient to PS, tagged with current local step ----
        response = await ps.push_grad(
            worker_id=worker_id,
            worker_step=state.local_step,
            grad_payload=payload,
            batch_indices=indices,
            delay=delay,
        )

        # ---- Apply PS response to local model ----
        if response["mode"] == "delta":
            # Fast-forward: add delta to current flat params
            current = flat_params(model)
            updated = current + response["delta"]
            load_flat_params(model, updated, device)
            state.local_step = ps.ps_step

        elif response["mode"] == "full":
            # Full sync (first contact, or extreme laggard)
            load_flat_params(model, response["flat"], device)
            state.local_step = ps.ps_step
            # Flush error buffer on full sync — stale residuals are garbage now
            state.error_buf = None

        await asyncio.sleep(0)

    print(f"[W{worker_id}] done. batches={state.batch_count}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_delay(dist: str, param: float) -> float:
    if dist == "exp":
        return random.expovariate(max(param, 1e-9))
    return float(np.random.poisson(param))


def get_datasets(data_root: str):
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.CIFAR10(data_root, train=True,
                                download=True, transform=None)
    val_ds   = datasets.CIFAR10(data_root, train=False,
                                download=True, transform=transform_val)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            num_workers=2, pin_memory=True)
    return train_ds, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(cfg: SimConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    codec = RHTCodec(bits=cfg.compress_bits) if cfg.compress else None

    run_name = cfg.wandb_run_name or (
        f"N{cfg.num_workers}"
        + (f"_rht{cfg.compress_bits}b" if cfg.compress else "_fp32")
        + f"_{cfg.delay_dist}_p{cfg.delay_param}"
    )
    wandb.init(
        project=cfg.wandb_project,
        name=run_name,
        config={
            "num_workers":   cfg.num_workers,
            "delay_dist":    cfg.delay_dist,
            "delay_param":   cfg.delay_param,
            "epochs":        cfg.epochs,
            "batch_size":    cfg.batch_size,
            "lr":            cfg.lr,
            "momentum":      cfg.momentum,
            "weight_decay":  cfg.weight_decay,
            "compress":      cfg.compress,
            "compress_bits": cfg.compress_bits,
            "staleness_cap": f"3N={3*cfg.num_workers}",
            "seed":          cfg.seed,
        },
    )
    if codec:
        wandb.log({"merger/compression_ratio": codec.compression_ratio})
        print(f"RHT: {cfg.compress_bits}b → {codec.compression_ratio:.1f}× compression")

    n_gpus = torch.cuda.device_count()
    print(f"GPUs: {n_gpus}")

    def pick_device(i: int) -> torch.device:
        return torch.device(f"cuda:{i % n_gpus}") if n_gpus > 0 \
               else torch.device("cpu")

    ps_device = pick_device(0)

    print("Building PS model...")
    ps_model = build_model().to(ps_device)
    ps_optim = optim.SGD(
        ps_model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    train_ds, val_loader = get_datasets(cfg.data_root)
    print(f"Train: {len(train_ds)} samples")

    distributor = BatchDistributor(
        n_samples=len(train_ds),
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        seed=cfg.seed,
    )

    ps = ParameterServer(
        model=ps_model,
        optimizer=ps_optim,
        val_loader=val_loader,
        eval_device=ps_device,
        distributor=distributor,
        num_workers=cfg.num_workers,
        codec=codec,
    )

    # Workers start with PS weights
    worker_models = []
    ps_flat = flat_params(ps_model)
    for i in range(cfg.num_workers):
        m = build_model()
        load_flat_params(m, ps_flat.clone(), torch.device("cpu"))
        worker_models.append(m)

    print(
        f"\nStarting:"
        f"\n  workers     = {cfg.num_workers}"
        f"\n  staleness   = {ps.max_lag} steps (3×N)"
        f"\n  compress    = {'RHT ' + str(cfg.compress_bits) + 'b' if cfg.compress else 'off'}"
        f"\n  optimizer   = SGD lr={cfg.lr} mom={cfg.momentum}"
        f"\n  delay       = {cfg.delay_dist}({cfg.delay_param})\n"
    )

    t0 = time.time()

    tasks = [
        asyncio.create_task(
            worker_loop(
                worker_id=i,
                model=worker_models[i],
                device=pick_device(i),
                delay_dist=cfg.delay_dist,
                delay_param=cfg.delay_param,
                ps=ps,
                distributor=distributor,
                train_ds=train_ds,
                codec=codec,
            )
        )
        for i in range(cfg.num_workers)
    ]

    await asyncio.gather(*tasks)

    elapsed = time.time() - t0
    total_rej = sum(ps._rejections.values())
    print(f"\nDone in {elapsed:.1f}s")
    print(f"PS steps:   {ps.ps_step}")
    print(f"Rejections: {total_rej} "
          f"{[ps._rejections.get(i, 0) for i in range(cfg.num_workers)]}")

    wandb.log({
        "summary/ps_steps":         ps.ps_step,
        "summary/total_rejections": total_rej,
        "summary/wall_time_s":      elapsed,
    })
    wandb.finish()


def parse_args() -> SimConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--num_workers",    type=int,   default=2)
    p.add_argument("--delay_dist",     choices=["exp", "poisson"], default="exp")
    p.add_argument("--delay_param",    type=float, default=0.3)
    p.add_argument("--epochs",         type=int,   default=3)
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--momentum",       type=float, default=0.9)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--data_root",      type=str,   default="./data")
    p.add_argument("--wandb_project",  type=str,   default="async-sgd-sim")
    p.add_argument("--wandb_run_name", type=str,   default=None)
    p.add_argument("--no-compress",    action="store_false", dest="compress")
    p.add_argument("--compress-bits",  type=int,   default=8,
                   choices=[2, 4, 8],  dest="compress_bits")
    p.set_defaults(compress=True)
    return SimConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
