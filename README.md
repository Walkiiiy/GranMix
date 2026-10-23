# GranMix

Code for GranMix, including training, checkpoint resume, and evaluation.

## Structure

```text
code/
  main.py           Command-line entry
  diagnosis.py      Partition Diagnosis (Section 3, Algorithm 1)
  refinement.py     Partition Refinement (Section 4, Algorithm 2)
  handoff.py        Mixture Handoff (Section 5, Algorithm 3)
  train.py          Model, optimizer, training, checkpoints, and evaluation
  data.py           Reference tree, corpus, and sampling
  config.py         Experiment configuration
  utils.py          Asset checks and serialization
  requirements.txt
configs/            Six main-experiment configurations
assets/             Pythia-160M configuration, tokenizer, and fixed tree
```

## Environment

Use Python 3.10. For GPU training, use the CUDA 12.1 build of PyTorch 2.2.1.

```bash
cd code
pip install -r requirements.txt
```

## Data

Pass the tokenized corpus root with `--data`. Each domain contains
`train.bin`, `val.bin`, and `test.bin` as uint16 token streams.
Tokenization and row order must match `assets/identity.json` and the bundled
tree labels. Large corpus files and trained checkpoints are not included.

The model is initialized from the bundled Pythia configuration.
Cases: `ase`, `ghc4`, `bse`, `abse`, `ccghw`, and `seven`.

## Usage

Run from `code/`:

```bash
python main.py inspect --case ase
python main.py preflight --case ase --data /path/to/corpus --full-hash

CUDA_VISIBLE_DEVICES=0 python main.py train \
  --case ase --data /path/to/corpus \
  --output /path/to/new_run --evaluate-test

python main.py train --case ase --data /path/to/corpus \
  --resume /path/to/new_run/checkpoints/step_000250.pt \
  --output /path/to/resumed_run --evaluate-test

python main.py evaluate \
  --checkpoint /path/to/resumed_run/checkpoints/step_005000.pt \
  --data /path/to/corpus --split test --output /path/to/test.json
```

GranMix is the default; use `--arm uniform` for the matched control.
Outputs must use new paths. For a partial run, add `--end 250 --skip-evaluation`.
Resume at a complete GranMix window boundary using a checkpoint from this code.

The five compact cases use 5,000 nominal updates; `seven` uses 40,000 nominal
and 39,439 actual updates. Each configuration specifies one seed.
Training supports one CPU/CUDA device with gradient checkpointing disabled.
Use `--allow-version-drift` for a different runtime; data checks still apply.

`python main.py smoke --output /path/to/new_smoke` runs a small synthetic CPU
experiment.
