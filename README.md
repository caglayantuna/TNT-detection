# TNT candidate detection for confocal z-stacks

Finds candidate tunneling nanotubes (TNTs) in 3D confocal stacks and writes them as an
[Icy](https://icy.bioimageanalysis.org/) ManualTNT file. The output is a ranked list for review, not a
validated detector.

## Files

| File | What it is |
|---|---|
| `run_tnt_detection_pipeline.py` | Runs the pipeline on one image or a folder. |
| `v4_nd2_wt006.ipynb` | Step-by-step notebook with figures on one Nikon stack. |

## Usage

```bash
/users/ctuna/microscopy/bin/python run_tnt_detection_pipeline.py 400k-sh-wt002
/users/ctuna/microscopy/bin/python run_tnt_detection_pipeline.py /path/to/folder
```

Input: an image name, a `.czi` or `.nd2` path, or a folder (every image inside is processed).

| Option | Meaning |
|---|---|
| `--top N` | Keep the N best candidates per image |
| `--out DIR` | Output folder (default `tnt_candidates/` next to the image) |
| `--sidecar` | Also write `<name>.xml` next to the image (an existing file is kept as a `.bak` copy) |
| `--device` | `mps` (default), `cuda` or `cpu` |

## Output

- `<name>.xml`: Icy ManualTNT file, one marker per cell and one line per candidate, best first.
- `<name>_candidates.csv`: one row per candidate, with rank, the two cells, the best plane, end points and scores.

## Opening in Icy

The output folder holds a link to the image next to the candidate `.xml`. Open the linked image in Icy.
In edit mode (Space), point at a candidate and press Delete (fn + Delete on a Mac) to remove it.

## Method

1. Drop a projection plane if present, and find the substrate.
2. Segment cells with Cellpose-SAM on the membrane channel.
3. Detect lines above the substrate with a noise-calibrated filter and the FAST F-actin network.
4. Keep lines that cross a gap and join two different cells.
5. Rank each cell pair by the FAST filopodia score along its best line.

## Results

- Nikon, 8 expert-marked TNTs: 6 in the top 20 of their stack, 5 in the top 10.
- Zeiss, annotated cell pairs: 18 of 51 in the top 20 per stack.
- Dense cultures: 20 of 95 annotated pairs in the top 20 on held-out stacks. Expect many false candidates.

## Requirements

- Python with `numpy`, `pandas`, `scipy`, `scikit-image`, `scikit-learn`, `matplotlib`, `nd2`,
  `aicspylibczi`, `cellpose` (v4 or later), `torch`.
- FAST model files (`fast_model.pth`, `model.py`) from BioImage.IO `reliable-waterbuffalo`, placed in
  `experiments_v4/models/fast_actin/`.
- Helper modules from the project folders `experiments/`, `experiments_hough_v2/`, `experiments_hough_v3/`
  and `experiments_v4/`.
