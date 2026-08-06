# Allen Cell Segmentation (optional / legacy backend)

This directory vendors the Allen Institute **`aicssegmentation`** classic
segmenter and a few thin driver scripts. It is an **optional** alternative to
the default Cellpose-SAM backend, useful for nucleus / cytoplasm / filament /
punctate-structure (e.g. mitochondria) segmentation with classic image-processing
workflows (no pretrained model required).

> The default MorphAgent segmentation backend is **Cellpose-SAM**, which lives in
> the unified `morphagent` environment and needs none of this. Only set up this
> backend if you specifically want the Allen classic workflows.

## 1. Installation

The Allen stack depends on an old, frozen scientific stack (Python 3.6) that
cannot be merged into the modern `morphagent` environment, so it lives in its
own environment.

```bash
# from the repo root
conda env create -f envs/environment_allen.yml   # creates morphagent_allen
conda activate morphagent_allen
pip install -e segmentation_allen                 # install vendored aicssegmentation
python segmentation_allen/check_installation.py   # verify
```

To make MorphAgent use this backend for segmentation:

```bash
export SEGMENTATION_CONDA_ENV=morphagent_allen
```

## 2. Driver scripts

All scripts take input/output paths as command-line arguments (run with `-h`
for the full list). They are examples you can adapt to your data.

| Script | Purpose |
|--------|---------|
| `run_segment_image_tif.py` | Nucleus + cytoplasm segmentation of a multi-channel TIFF |
| `run_segment_mitochondria.py` | Mitochondria / punctate-structure segmentation |
| `segmentation_pipeline.py` | Batch nucleus + cytoplasm segmentation over a folder |
| `test_single_image.py` | Smoke test on a single image |
| `segment_tau_mip_cell_bundle.py` | Example: cell + bundle segmentation on a MIP |

Example:

```bash
conda activate morphagent_allen
python segmentation_allen/run_segment_image_tif.py \
    --image /path/to/image.tif \
    --output /path/to/output_dir
```

## 3. Channels

The classic workflows expect you to point them at the right channel for each
structure. The mapping is **dataset-specific** — pass the appropriate channel
index/name for your images (see each script's `-h`). For example, a 3-channel
image might use a DNA channel for the nucleus and an actin/membrane channel for
the cytoplasm; adjust to match your acquisition.

## 4. Output

- **Segmentation masks**: TIFF, binary (0 / 255).
- **Visualization**: PNG, original image overlaid with the segmentation.

## 5. Parameter tuning

The classic workflows expose a few parameters you can edit inside the driver
scripts (e.g. `segmentation_pipeline.py`):

```python
# Nucleus
intensity_norm_param = [0.5, 15]   # intensity normalization
gaussian_smoothing_sigma = 1.0     # Gaussian smoothing
object_minArea = 10                # minimum nucleus area
min_size = 50                      # minimum object size (postprocessing)

# Cytoplasm
gaussian_smoothing_sigma = 1.5     # cytoplasm needs more smoothing
object_minArea = 100               # minimum cytoplasm area
min_size = 200                     # minimum object size (postprocessing)
dilation_radius = 5                # nucleus dilation radius
```

## 6. Troubleshooting

**`ModuleNotFoundError: No module named 'aicssegmentation'`**

Make sure the environment is active and the vendored package is installed:

```bash
conda activate morphagent_allen
pip install -e segmentation_allen
python segmentation_allen/check_installation.py
```

**Poor segmentation results** — tune the parameters above (normalization,
`object_minArea`, `gaussian_smoothing_sigma`) to match your image contrast and
object sizes.

## Related

- `aicssegmentation/` — vendored Allen classic segmenter (see its own `README.md`).
- `check_installation.py` — verifies the backend is importable.
