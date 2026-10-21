# LRA Retrieval: controlled MHA / Alam HRR / direct HRR comparison

## Workspace findings

- No existing LRA loader or paired-document classifier was found in Zoology.
- `zoology/zoology/config.py`: `ModuleConfig.instantiate` imports a dotted class name; no central registration is required.
- `zoology/zoology/experiments/models_repo.py`: existing mixer configuration catalogue.
- `zoology/zoology/launch.py`: loads a Python file containing `configs`, then invokes the existing trainer.
- `zoology/zoology/data/utils.py`: synthetic task assembly and caching.
- `zoology/zoology/model.py` and `train.py`: vocabulary prediction and token-level loss, rather than paired-document classification.
- `zoology/zoology/mixers/hrrformer.py`: `HRRFormerMixer` implements the cosine/sequence-softmax value weighting; `MyHRRFormerMixer` directly returns the retrieved vector after head merging and output projection.
- `zoology/zoology/mixers/attention.py`: current MHA applies a hard-coded causal mask. Retrieval needs a noncausal comparison with the global HRR memories.

The separate runner is the smallest self-contained addition that avoids changing the MQAR trainer and its experiments. Do not launch it through `zoology.launch`.

## Exact PowerShell commands

```powershell
Set-Location C:\Users\hamze\VSAworkspace
$env:PYTHONPATH = 'C:\Users\hamze\VSAworkspace\zoology'
& .\.venv\Scripts\python.exe .\lra_retrieval.py smoke
& .\.venv\Scripts\python.exe .\prepare_lra.py
& .\.venv\Scripts\python.exe .\lra_retrieval.py train --mixers mha alam direct --seeds 0 1 2
```

Preparation downloads the three original AAN TSVs from the OpenNLPLab/lra mirror at pinned revision `6cb133c44f406da743661488df9e977b87d98775`, verifies each SHA-256 against the mirror's published LFS object checksum, and writes one shared disk-backed token cache. The original Google archive endpoint currently returns HTTP 403. Raw TSV downloads total 10.6 GB; allow at least 16 GB of free disk space for files and caches. Interrupted downloads use `.tsv.partial` files and resume when rerun. The mirror is third-party: checksums establish integrity against that pinned mirror revision, not independently proven identity to the now-inaccessible Google archive. Split filenames and membership are retained without resplitting. No TensorFlow, JAX, new package installation, or W&B account is required. CUDA-enabled PyTorch is available in the existing environment.

If you already have the official release, skip the download:

```powershell
& .\.venv\Scripts\python.exe .\lra_retrieval.py prepare --source 'C:\path\to\tsv_data'
```

Individual runs use the same settings and seeds:

```powershell
& .\.venv\Scripts\python.exe .\lra_retrieval.py train --mixers alam --seeds 0 1 2
& .\.venv\Scripts\python.exe .\lra_retrieval.py train --mixers direct --seeds 0 1 2
& .\.venv\Scripts\python.exe .\lra_retrieval.py train --mixers mha --seeds 0 1 2
```

Run either the combined command or the individual commands. Existing run directories cause an error to protect results. For a fresh repeat, supply a different `--output`, e.g. `--output runs/lra_repeat2`. There is no interrupted-run resume support.

## Shared experiment settings

| Setting | All three mixers |
|---|---|
| Dataset | Official AAN `new_aan_pairs.train/eval/test.tsv` |
| Tokenizer | UTF-8 bytes + 1, padding ID 0, vocabulary 257 |
| Sequence length | 4,000 bytes per document, two separately encoded documents |
| Embedding / heads / layers | 128 / 4 / 4 |
| Epochs | 20, no early stopping |
| Seeds | 0, 1, 2 |
| Batch size | 4 pairs, including final incomplete batches |
| Optimizer | Adam, default betas (0.9, 0.999), epsilon 1e-8, zero weight decay |
| Learning rate | 1e-3, continuous decay factor 0.9 per training epoch, floor 1e-5 |
| Dropout | 0.1 on embedding and block outputs; no attention-weight dropout |
| Loss | Binary cross entropy via two logits, label smoothing 0.1 |
| Positions | Fixed sinusoidal embeddings |
| Initialization | Normal weights with standard deviation 0.02 for every Linear/Embedding; zero biases and padding embedding |
| Blocks | Shared normalization before both mixer and feed-forward sublayers, residual connections, and final encoder normalization |
| Classifier | Shared encoder, masked mean pooling; concatenate a, b, a*b, a-b; 128-unit GELU head |
| Checkpoint selection | Highest validation accuracy; final test evaluated once on selected checkpoint |

`Config` in `lra_retrieval.py` is the single settings source. `RetrievalMixer` is the local registration point for `mha`, `alam`, and `direct`. All three reuse the same local HRR class projection layout, including bias-free Q/K/V/O, ensuring identical parameter counts and initialization. MHA uses PyTorch scaled dot-product attention with a padding mask and no causal mask. Both HRR modes reuse the workspace's binding, unbinding, head split, and merge methods. The runner reproduces the respective readout formulas with explicit masks rather than invoking their unmasked `forward` methods.

Padding contributes neither to HRR memory nor to MHA keys; padding query outputs and block states are zeroed. Pooling excludes padding. Every run reads the same cache, validates its SHA-256 hashes, and recreates the same seeded data shuffle generator. Same seeds produce identical starting model state across mixers. GPU kernels may still be nondeterministic.

To reduce GPU memory, use `--batch-size 2` on the combined command (or every individual command), with a fresh output directory. This changes the shared batch size and optimizer step schedule; do not compare it with batch-size-4 runs as if the settings matched. Default training is full float32. MHA uses an efficient SDPA kernel when supported; fallback kernels may need more memory.

## Results and interpretation

Each `runs/lra_retrieval/<mixer>_seed<seed>/` contains `config.json` with split hashes and settings, `history.jsonl` with all train/validation epochs, `best.pt`, and `result.json` with validation and held-out test accuracy. Accuracy is a fraction between 0 and 1. Compare test accuracy across matched seeds and report the mean and standard deviation. A full dataset training run has not yet been validated.

This is a controlled PyTorch ablation, not an exact reproduction of the authors' reported scores. Differences include PyTorch initialization, consistent removal of padding from memory and pooling, complete evaluation batches, and single-device batch size. The authors' multi-GPU entry point multiplies batch size 4 by device count; the local value 4 is a practical shared single-device default. Their public Retrieval loader also returns two datasets while the multi-GPU script expects three, so it cannot be copied directly as a working experiment.

The runner uses normalization before each mixer, as in Zoology, for all three variants. Full-length testing found direct HRR overflowed with the authors' unnormalized attention residual layout, even with small initialization. This shared normalization is an additional difference from the paper model; do not interpret scores as a faithful reproduction without a separate architecture study.

The previous conversation's paper scores are reference targets, not guarantees. First inspect Alam learning curves and test accuracy before attributing any improvement to direct retrieval. LRA Retrieval measures binary document matching, so it should not be expected to replicate MQAR's near-perfect MHA accuracy.

Sources: [official HRRFormer implementation](https://github.com/FutureComputing4AI/Hrrformer), [authors' Retrieval model](https://github.com/FutureComputing4AI/Hrrformer/blob/master/retrieval/hrrformer_mgpu.py), [official LRA preprocessing](https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/matching/input_pipeline.py), [paper](https://proceedings.mlr.press/v202/alam23a.html).
