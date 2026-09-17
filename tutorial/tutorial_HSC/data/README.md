Images are **not** bundled.

Expected layout after you copy a local dataset:

```
data/dataset/<sample_id>/
  image.tif
  slices/*.png
  segmentation/mito.tif
  segmentation/cell.tif      # optional
  segmentation/nucleus.tif   # optional
```

`data/outputs/` is written by the notebooks (smoke CSVs and replayed figures).
