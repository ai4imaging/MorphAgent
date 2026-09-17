# data/

| Path | Role |
|------|------|
| `dataset/` | BBBC021 sample folders (`image.tif` + `slices/` + `segmentation/`). Gitignored. |
| `outputs/` | Plots and smoke CSVs written by the notebooks. Gitignored. |
| `evaluation/` | Optional large feature matrices for a full recompute of notebook 03. Not required. |

Download `dataset/` with `notebook/01_setup_environment_and_data.ipynb` or:

```bash
python code/download_dataset.py
```

Source: [Zenodo record 22763120](https://zenodo.org/records/22763120) (DOI [10.5281/zenodo.22763120](https://doi.org/10.5281/zenodo.22763120)). The archive is uploaded as `dataset.zip.part_*`; the helper concatenates those parts (`cat dataset.zip.part_* > dataset.zip`) and extracts them. If `data/dataset/<sample_id>/image.tif` already exists, skip the download.
