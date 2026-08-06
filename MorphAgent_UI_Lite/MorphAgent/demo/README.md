# MorphAgent UI Lite Demo

The bundled demo contains 10 Tau neuron microscopy samples:

- `WT_1`–`WT_5`: wild type
- `MU_1`–`MU_5`: mutant

It is configured for a short one-round run and includes a completed result that can be browsed without an API key.

## Contents

```text
demo/
├── data/
│   ├── dataset/              # images, slices, and existing masks
│   ├── metadata.csv          # sample labels
│   └── results/
│       └── completed_demo_run/
├── precomputed/              # prepared knowledge summaries
└── morphagent_demo.ipynb     # optional notebook
```

## Run in the UI

1. Start MorphAgent UI Lite.
2. Click **Load demo dataset**.
3. Enter compatible LLM and VLM API settings.
4. Click **Run MorphAgent**.
5. Open **Features** and **Evidence** after completion.

The demo reuses its bundled segmentation masks and does not require automatic segmentation.

## Custom dataset layout

```text
dataset/
├── sample_1/
│   ├── image.tif
│   └── segmentation/         # optional
└── sample_2/
    └── image.png
```

At least five samples are recommended for useful validation. Smaller datasets show a warning but can still run.

See [`../../README_LITE.md`](../../README_LITE.md) for installation and feature-reuse instructions.
