# Attention Based Optimization for 3D Shape Registration

[![DOI](https://img.shields.io/badge/DOI-10.1111%2Fcgf.70525-blue)](https://doi.org/10.1111/cgf.70525)

Official implementation of:

> **Attention Based Optimization for 3D Shape Registration**
> A. Riva, L. Olearo, S. Melzi
> *Computer Graphics Forum* (Proc. Eurographics Symposium on Geometry Processing 2026)
> DOI: [10.1111/cgf.70525](https://doi.org/10.1111/cgf.70525)

![Pipeline overview: the two input shapes A and B are processed by a self-attention feature extractor, then a cross-attention point matcher is applied in both directions and the resulting attention weights are composed into the final correspondence/registration.](img/architecture.png)

## Abstract

Transformers are sequence-to-sequence architectures originally designed for structurally rigid, order-sensitive data such as text and images. At their core lies the attention mechanism, which is permutation-equivariant and relies on computing token-to-token relationships, and which has also been applied to 3D geometry tasks such as generation, segmentation, classification, shape matching, and registration. While existing methods use transformers as traditional learners, we instead reinterpret the transformer itself as an **optimization pipeline for shape correspondence**: by fitting the model directly to a single shape pair, our approach removes the need for large training datasets and yields a category-agnostic solution. Self-attention acts as a feature extractor, while cross-attention acts as a matcher. We exploit the cross-attention weights *directly* as the permutation/correspondence matrix, without any additional decoding. The result is a robust, interpretable, training-data-free pipeline for non-rigid shape matching and registration, requiring only a handful of landmarks (~5) to ground the optimization.

## Installation

The project targets Python 3.10+ and was tested with PyTorch 2.x. We recommend working inside a virtual environment.

1. **Clone the repository together with its submodule** ([`meshtorch`](https://github.com/ariva00/meshtorch)):

   ```bash
   git clone --recursive git@github.com:ariva00/Attention4ShapeRegistration.git
   cd Attention4ShapeRegistration
   ```

   If you already cloned without `--recursive`, fetch the submodule with:

   ```bash
   git submodule update --init --recursive
   ```

2. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the base requirements**:

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Install PyRMT**, needed only for `--rmt` (the Rematching-based resampling described in Section 4.6.2). It must be installed from the `python-binding` branch of the original repository, **not** any package index:

   ```bash
   git clone --branch python-binding https://github.com/filthynobleman/rematching.git
   pip install ./rematching
   ```

   Refer to that repository's own README for any platform-specific build prerequisites (it builds a C++ extension).

5. **(Optional) Install [geomfum](https://github.com/3diglab/geomfum)**, needed only for `--diffusionnet` (Section 6.2). Install from source rather than from PyPI, since the project relies on the latest `main` branch:

   ```bash
   git clone https://github.com/3diglab/geomfum.git
   pip install ./geomfum
   ```

   Steps 4 and 5 are both optional: the rest of the pipeline runs fine without them, and
   passing `--rmt`/`--diffusionnet` without the corresponding package installed raises a
   clear error telling you to install it.

## Datasets

Datasets are expected under `dataset/` (configurable via `--path-data`), with the following layout per dataset:

```
dataset/
├── MPI-FAUST/
│   ├── training/registrations/*.ply
│   └── original_to_sym_map.txt
├── SHREC20b_lores/
│   ├── models/*.obj
│   └── test-sets/test-set{0..4}.txt
├── SHREC20b_lores_gts/*.mat
├── SMAL_r/
│   ├── off/*.off
│   ├── corres/*.vts
│   └── test_cat.txt
└── TOPKIDS/
    ├── off/*.off
    └── corres/*_ref.vts
```

## Running

The method fits the model to a single pair of shapes at a time (it is an optimization pipeline, not a trained model that generalizes across pairs), so `train.py` both fits and evaluates one couple, and the fitted weights can later be reloaded with `test.py`.

### Fit + evaluate a single couple

```bash
python train.py \
    --dataset faust \
    --path-data dataset/ \
    --couple-names tr_reg_080 tr_reg_081 \
    --landmarks 5 \
    --run-name my_run
```

This trains (pretraining + matching, Section 4.5) and evaluates the couple, saving the model checkpoint, training/pretraining logs, and the run's CSVs under `./models/my_run/` (configurable via `--path-model`).

### Evaluate a previously fitted checkpoint

```bash
python test.py \
    --dataset faust \
    --path-data dataset/ \
    --couple-names tr_reg_080 tr_reg_081 \
    --landmarks 5 \
    --run-name my_run
```

Run `python train.py --help` for the full list of options (architecture size, ablations such as `--self-only`/`--matcher-only`/`--no-symmetric`, resampling/Rematching settings, DiffusionNet/Diff3f input features, etc.).

### Reproducing the paper's experiments

The `experiments/` scripts loop `train.py`/`test.py` over every couple of a benchmark and log results to a single CSV:

```bash
python -m experiments.faust     # MPI-FAUST
python -m experiments.shrec20   # SHREC'20 (non-isometric, topological noise)
python -m experiments.smal_r    # SMAL-R (animals)
python -m experiments.topkids   # TOPKIDS
```

Pass `--test` to evaluate already-fitted models instead of fitting new ones, and any of the flags from `train.py` to override the defaults (e.g. landmark count, architecture size, ablations).

## Citation

If you use this code, please cite:

```bibtex
@article{riva2026attention,
  title     = {Attention Based Optimization for 3D Shape Registration},
  author    = {Riva, A. and Olearo, L. and Melzi, S.},
  journal   = {Computer Graphics Forum},
  year      = {2026},
  doi       = {10.1111/cgf.70525}
}
```

## License

This project is released under the MIT License, see [LICENCE](LICENCE).
