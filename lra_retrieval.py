"""Controlled PyTorch LRA AAN Retrieval comparison using local Zoology HRR classes.

Run --help for preparation, training, and smoke-test commands. This is a
controlled ablation, not a numerically exact reproduction of the JAX paper.
"""

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass(frozen=True)
class Config:
    length: int = 4000
    d_model: int = 128
    heads: int = 4
    layers: int = 4
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-3
    dropout: float = 0.1
    weight_decay: float = 0.0
    label_smoothing: float = 0.1


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(source, destination, length):
    """Preserve official train/eval/test membership; UTF-8 bytes + 1, pad=0."""
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "length": length,
        "tokenizer": "utf8-byte-plus-one",
        "splits": {},
    }

    for split, suffix in [
        ("train", "train"),
        ("valid", "eval"),
        ("test", "test"),
    ]:
        path = source / f"new_aan_pairs.{suffix}.tsv"

        if not path.is_file():
            raise FileNotFoundError(f"Missing official LRA file: {path}")

        count = 0

        with path.open(encoding="utf-8", newline="") as f:
            for line in f:
                fields = line.rstrip("\r\n").split("\t")

                if len(fields) != 5 or float(fields[0]) not in (0, 1):
                    raise ValueError(
                        f"Invalid TSV row {count + 1} in {path}"
                    )

                count += 1

        if count == 0:
            raise ValueError(f"Empty split: {path}")

        x = np.lib.format.open_memmap(
            destination / f"{split}_x.npy",
            mode="w+",
            dtype=np.uint16,
            shape=(count, 2, length),
        )

        y = np.lib.format.open_memmap(
            destination / f"{split}_y.npy",
            mode="w+",
            dtype=np.int64,
            shape=(count,),
        )

        x[:] = 0

        with path.open(encoding="utf-8", newline="") as f:
            for i, line in enumerate(f):
                label, _, _, text1, text2 = line.rstrip("\r\n").split("\t")
                y[i] = int(float(label))

                for j, text in enumerate((text1, text2)):
                    tokens = np.frombuffer(
                        text.encode("utf-8")[:10000][:length],
                        dtype=np.uint8,
                    )

                    x[i, j, : len(tokens)] = (
                        tokens.astype(np.uint16) + 1
                    )

        x.flush()
        y.flush()

        manifest["splits"][split] = {
            "count": count,
            "source": str(path.resolve()),
            "source_sha256": digest(path),
            "x_sha256": digest(destination / f"{split}_x.npy"),
            "y_sha256": digest(destination / f"{split}_y.npy"),
        }

        print(f"{split}: {count} pairs", flush=True)

        del x, y

    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )


class Pairs(Dataset):
    def __init__(self, directory, split, length):
        self.x = np.load(
            directory / f"{split}_x.npy",
            mmap_mode="r",
        )

        self.y = np.load(
            directory / f"{split}_y.npy",
            mmap_mode="r",
        )

        if self.x.shape != (len(self.y), 2, length):
            raise ValueError(
                "Cache shape disagrees with shared configuration"
            )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.x[i], dtype=torch.long),
            int(self.y[i]),
        )


class RetrievalMixer(nn.Module):
    """Same Q/K/V/O parameters for all mixers; consistent padding semantics."""

    def __init__(self, config, kind):
        super().__init__()

        from zoology.mixers.hrrformer import (
            HRRFormerMixer,
            MyHRRFormerMixer,
        )

        cls = (
            MyHRRFormerMixer
            if kind == "direct"
            else HRRFormerMixer
        )

        self.core = cls(
            config.d_model,
            num_heads=config.heads,
        )

        self.kind = kind

    def forward(self, x, mask):
        c = self.core

        q, k, v = [
            c.split_heads(proj(x))
            for proj in (c.query, c.key, c.value)
        ]

        valid = mask[:, None, :, None]

        if self.kind == "mha":
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=valid.transpose(-1, -2),
                dropout_p=0.0,
                is_causal=False,
            )

        else:
            memory = (
                c.binding(k, v) * valid
            ).sum(dim=-2, keepdim=True)

            retrieved = c.unbinding(
                memory,
                q,
            )

            if self.kind == "direct":
                out = retrieved

            else:
                similarity = F.cosine_similarity(
                    v,
                    retrieved,
                    dim=-1,
                ).unsqueeze(-1)

                weights = similarity.masked_fill(
                    ~valid,
                    float("-inf"),
                ).softmax(dim=-2)

                out = weights * v

        return (
            c.output(c.merge_heads(out))
            * mask.unsqueeze(-1)
        )


class Block(nn.Module):
    def __init__(self, config, kind):
        super().__init__()

        self.mixer = RetrievalMixer(
            config,
            kind,
        )

        self.mixer_norm = nn.LayerNorm(
            config.d_model,
            eps=1e-6,
        )

        self.norm = nn.LayerNorm(
            config.d_model,
            eps=1e-6,
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                config.d_model,
                config.d_model // 2,
            ),
            nn.GELU(),
            nn.Linear(
                config.d_model // 2,
                config.d_model,
                bias=False,
            ),
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

    def forward(self, x, mask):
        x = x + self.dropout(
            self.mixer(
                self.mixer_norm(x),
                mask,
            )
        )

        return (
            x
            + self.dropout(
                self.ffn(
                    self.norm(x)
                )
            )
        ) * mask.unsqueeze(-1)


class PairClassifier(nn.Module):
    def __init__(self, config, kind):
        super().__init__()

        self.embedding = nn.Embedding(
            257,
            config.d_model,
            padding_idx=0,
        )

        positions = torch.arange(
            config.length
        ).unsqueeze(1)

        frequencies = torch.exp(
            torch.arange(
                0,
                config.d_model,
                2,
            )
            * (
                -math.log(10000)
                / config.d_model
            )
        )

        pe = torch.zeros(
            config.length,
            config.d_model,
        )

        pe[:, 0::2] = torch.sin(
            positions * frequencies
        )

        pe[:, 1::2] = torch.cos(
            positions * frequencies
        )

        self.register_buffer(
            "positions",
            pe,
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

        self.norm = nn.LayerNorm(
            config.d_model,
            eps=1e-6,
        )

        self.blocks = nn.ModuleList(
            [
                Block(config, kind)
                for _ in range(config.layers)
            ]
        )

        self.final_norm = nn.LayerNorm(
            config.d_model,
            eps=1e-6,
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                4 * config.d_model,
                config.d_model,
            ),
            nn.GELU(),
            nn.Linear(
                config.d_model,
                2,
            ),
        )

        # Match Zoology's small-weight initialization across all mixers.
        # Default Linear initialization can overflow stacked direct-HRR reads.
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(
            module,
            (nn.Linear, nn.Embedding),
        ):
            nn.init.normal_(
                module.weight,
                std=0.02,
            )

            if (
                isinstance(module, nn.Linear)
                and module.bias is not None
            ):
                nn.init.zeros_(
                    module.bias
                )

            if (
                isinstance(module, nn.Embedding)
                and module.padding_idx is not None
            ):
                with torch.no_grad():
                    module.weight[
                        module.padding_idx
                    ].zero_()

    def encode(self, ids):
        mask = ids != 0

        if not mask.any(dim=1).all():
            raise ValueError(
                "Empty document after tokenization"
            )

        x = self.norm(
            self.dropout(
                self.embedding(ids)
                + self.positions[: ids.shape[1]]
            )
        )

        x = x * mask.unsqueeze(-1)

        for block in self.blocks:
            x = block(
                x,
                mask,
            )

        x = (
            self.final_norm(x)
            * mask.unsqueeze(-1)
        )

        return (
            x.sum(dim=1)
            / mask.sum(
                dim=1,
                keepdim=True,
            )
        )

    def forward(self, pairs):
        a = self.encode(
            pairs[:, 0]
        )

        b = self.encode(
            pairs[:, 1]
        )

        features = torch.cat(
            [
                a,
                b,
                a * b,
                a - b,
            ],
            dim=-1,
        )

        return self.classifier(
            features
        )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fixed_subset(dataset, n, seed):
    """Select a deterministic random subset shared by every model."""

    if n is None or n >= len(dataset):
        return dataset

    if n <= 0:
        raise ValueError(
            "Subset size must be positive"
        )

    rng = np.random.default_rng(
        seed
    )

    indices = rng.choice(
        len(dataset),
        size=n,
        replace=False,
    )

    return Subset(
        dataset,
        indices.tolist(),
    )


def run_epoch(
    model,
    loader,
    device,
    config,
    optimizer=None,
    scheduler=None,
):
    model.train(
        optimizer is not None
    )

    total = 0
    correct = 0
    loss_sum = 0.0

    description = (
        "Training"
        if optimizer is not None
        else "Evaluation"
    )

    with torch.set_grad_enabled(
        optimizer is not None
    ):
        for pairs, labels in tqdm(
            loader,
            desc=description,
        ):
            pairs = pairs.to(device)
            labels = labels.to(device)

            logits = model(
                pairs
            )

            loss = F.cross_entropy(
                logits,
                labels,
                label_smoothing=config.label_smoothing,
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Nonfinite loss; do not compare this run"
                )

            if optimizer is not None:
                optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()
                optimizer.step()
                scheduler.step()

            total += len(labels)

            correct += (
                logits.argmax(-1)
                == labels
            ).sum().item()

            loss_sum += (
                loss.item()
                * len(labels)
            )

    return {
        "loss": loss_sum / total,
        "accuracy": correct / total,
        "examples": total,
    }


def train(args, config):
    manifest = json.loads(
        (
            args.data
            / "manifest.json"
        ).read_text()
    )

    if manifest["length"] != config.length:
        raise ValueError(
            "Prepared length differs from experiment length"
        )

    for split in (
        "train",
        "valid",
        "test",
    ):
        for field in (
            "x",
            "y",
        ):
            path = (
                args.data
                / f"{split}_{field}.npy"
            )

            expected = (
                manifest["splits"][split]
                [f"{field}_sha256"]
            )

            if digest(path) != expected:
                raise ValueError(
                    f"Cache changed: {split}_{field}"
                )

    datasets = {
        s: Pairs(
            args.data,
            s,
            config.length,
        )
        for s in (
            "train",
            "valid",
            "test",
        )
    }

    datasets["train"] = fixed_subset(
        datasets["train"],
        args.train_examples,
        12345,
    )

    datasets["valid"] = fixed_subset(
        datasets["valid"],
        args.valid_examples,
        23456,
    )

    datasets["test"] = fixed_subset(
        datasets["test"],
        args.test_examples,
        34567,
    )

    print(
        "Using examples:",
        f"train={len(datasets['train'])},",
        f"valid={len(datasets['valid'])},",
        f"test={len(datasets['test'])}",
        flush=True,
    )

    for seed in args.seeds:
        for kind in args.mixers:
            seed_all(seed)

            model = PairClassifier(
                config,
                kind,
            ).to(args.device)

            generator = (
                torch.Generator()
                .manual_seed(seed)
            )

            loaders = {
                s: DataLoader(
                    d,
                    batch_size=config.batch_size,
                    shuffle=s == "train",
                    generator=(
                        generator
                        if s == "train"
                        else None
                    ),
                    num_workers=0,
                )
                for s, d
                in datasets.items()
            }

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

            steps = len(
                loaders["train"]
            )

            scheduler = (
                torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lambda step: max(
                        1e-5
                        / config.learning_rate,
                        0.9
                        ** (
                            max(
                                0,
                                step - 1,
                            )
                            / steps
                        ),
                    ),
                )
            )

            out = (
                args.output
                / f"{kind}_seed{seed}"
            )

            out.mkdir(
                parents=True,
                exist_ok=False,
            )

            run_config = {
                **asdict(config),
                "seed": seed,
                "mixer": kind,
                "optimizer": "Adam",
                "schedule": (
                    "exponential-0.9-per-epoch-floor-1e-5"
                ),
                "torch": torch.__version__,
                "dataset": manifest,
                "subsets": {
                    "train": len(
                        datasets["train"]
                    ),
                    "valid": len(
                        datasets["valid"]
                    ),
                    "test": len(
                        datasets["test"]
                    ),
                    "subset_seeds": {
                        "train": 12345,
                        "valid": 23456,
                        "test": 34567,
                    },
                },
            }

            (
                out
                / "config.json"
            ).write_text(
                json.dumps(
                    run_config,
                    indent=2,
                )
            )

            best = -1.0

            for epoch in range(
                1,
                config.epochs + 1,
            ):
                train_metrics = run_epoch(
                    model,
                    loaders["train"],
                    args.device,
                    config,
                    optimizer,
                    scheduler,
                )

                valid_metrics = run_epoch(
                    model,
                    loaders["valid"],
                    args.device,
                    config,
                )

                metrics = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "valid": valid_metrics,
                }

                if (
                    valid_metrics["accuracy"]
                    > best
                ):
                    best = (
                        valid_metrics[
                            "accuracy"
                        ]
                    )

                    torch.save(
                        model.state_dict(),
                        out / "best.pt",
                    )

                with (
                    out
                    / "history.jsonl"
                ).open("a") as f:
                    f.write(
                        json.dumps(
                            metrics
                        )
                        + "\n"
                    )

                print(
                    kind,
                    seed,
                    json.dumps(metrics),
                    flush=True,
                )

            model.load_state_dict(
                torch.load(
                    out / "best.pt",
                    map_location=args.device,
                    weights_only=True,
                )
            )

            result = {
                "best_valid_accuracy": best,
                "test": run_epoch(
                    model,
                    loaders["test"],
                    args.device,
                    config,
                ),
            }

            (
                out
                / "result.json"
            ).write_text(
                json.dumps(
                    result,
                    indent=2,
                )
            )

            print(
                kind,
                seed,
                result,
                flush=True,
            )

            del model
            del optimizer
            del scheduler


def smoke():
    config = Config(
        length=32,
        d_model=16,
        heads=2,
        layers=2,
    )

    pairs = torch.randint(
        1,
        257,
        (2, 2, 32),
    )

    pairs[:, :, 20:] = 0

    states = []

    for kind in (
        "mha",
        "alam",
        "direct",
    ):
        seed_all(0)

        model = PairClassifier(
            config,
            kind,
        ).eval()

        states.append(
            model.state_dict()
        )

        logits = model(
            pairs
        )

        F.cross_entropy(
            logits,
            torch.tensor([0, 1]),
        ).backward()

        assert (
            logits.shape
            == (2, 2)
        )

        assert torch.isfinite(
            logits
        ).all()

        assert all(
            p.grad is None
            or torch.isfinite(
                p.grad
            ).all()
            for p
            in model.parameters()
        )

        torch.testing.assert_close(
            logits,
            model(
                pairs[:, :, :20]
            ),
            rtol=1e-4,
            atol=1e-5,
        )

        print(
            f"{kind}: forward, backward, padding checks passed"
        )

    for name in states[0]:
        assert all(
            torch.equal(
                states[0][name],
                state[name],
            )
            for state
            in states[1:]
        ), name

    print(
        "All three models start with identical parameters for the same seed."
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "train",
            "smoke",
        ],
    )

    parser.add_argument(
        "--source",
        type=Path,
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "data/lra_retrieval"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/lra_retrieval"
        ),
    )

    parser.add_argument(
        "--mixers",
        nargs="+",
        choices=[
            "mha",
            "alam",
            "direct",
        ],
        default=[
            "alam",
            "direct",
        ],
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[
            0,
            1,
            2,
        ],
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--length",
        type=int,
        default=4000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--train-examples",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--valid-examples",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--test-examples",
        type=int,
        default=5000,
    )

    args = parser.parse_args()

    config = Config(
        length=args.length,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    if args.action == "prepare":
        if args.source is None:
            parser.error(
                "prepare requires --source pointing to the official tsv_data directory"
            )

        prepare(
            args.source,
            args.data,
            config.length,
        )

    elif args.action == "smoke":
        smoke()

    else:
        train(
            args,
            config,
        )


if __name__ == "__main__":
    main()