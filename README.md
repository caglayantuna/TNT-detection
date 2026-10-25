# TNT candidate detection for confocal z-stacks

This pipeline finds **candidate tunneling nanotubes (TNTs)** in 3D confocal stacks and writes them as
an [Icy](https://icy.bioimageanalysis.org/) ManualTNT annotation file. Biologists open the file, keep the
candidates that are TNTs, delete the rest, and add any that were missed.

The output is a **ranked list for review**. It is not a validated detector. Read the
[Results](#results) section before using it.

## Contents

| File | What it is |
|---|---|
| `run_tnt_detection_pipeline.py` | Command-line tool. Takes one image, or a folder of images, and writes the Icy file and a candidate table. |
| `v4_nd2_wt006.ipynb` | Notebook that shows the method step by step on one Nikon stack: cells, candidates, a review sheet with side views, and the overview figure. |

## Quick start

Run from the folder that contains the script:

```bash
/users/ctuna/microscopy/bin/python run_tnt_detection_pipeline.py 400k-sh-wt002
```

The input can be:

- an image name (`1`, `1.czi`, `400k-sh-wt002`), looked up in the two data folders,
- a path to a `.czi` or `.nd2` file,
- a folder. Every `.czi` and `.nd2` file directly inside it is processed. If one image fails, the error is
  printed and the others still run.

Options:

| Option | Meaning |
|---|---|
| `--top N` | Keep only the N best candidates per image. Default: all. |
| `--out DIR` | Output folder. Default: `tnt_candidates/` next to the image. |
| `--sidecar` | Also write `<name>.xml` next to the image, so Icy opens it with the image. An existing file is first renamed to `<name>.xml.bak-<timestamp>`. |
| `--no-link` | Do not put a link to the image in the output folder. |
| `--device` | Torch device: `mps` (default, Apple GPU), `cuda` or `cpu`. |

## Output

For each image the script writes two files in the output folder.

**`<name>.xml`**: an Icy document in the same format as the annotators' files.
- The `plugins.fab.manualtnt.ManualTNT` block holds one `<cell class="0" x= y=/>` marker for each detected
  cell, and one `<tnt x1= x2= y1= y2=/>` line for each candidate. Candidates are ordered best first.
- Coordinates are in full-resolution image pixels.
- Metadata and display settings are copied from an existing annotation file if one sits next to the image.

**`<name>_candidates.csv`**: one row per candidate, in the same order as the `<tnt>` lines.

| Column | Meaning |
|---|---|
| `rank` | Position in the list (1 = best). |
| `src` | Which detector found the line: `gap`, `fast`, or both. |
| `z` | Plane where the line is best seen. |
| `cell_a`, `cell_b` | The two cells the line joins (labels in the cell map). |
| `x0`, `y0`, `x1`, `y1` | End points, full-resolution pixels. |
| `filopodia` | FAST filopodia probability along the line. This is the ranking score. |
| `fibres` | FAST actin and stress-fibre probability along the line. |
| `contrast` | Line brightness above the dark background, in noise units. |
| `over_gap` | Share of the line's middle that lies outside every cell. |
| `length_um`, `elev_um` | Line length, and height of the line above the substrate. |

## Using the file in Icy

Icy loads the ManualTNT annotations from an `.xml` file with the same name as the image, placed next to it.
Two ways to open a candidate file:

1. **Default (recommended).** The script puts a link to the image inside the output folder, next to the
   candidate `.xml`, and copies the plugin's cached rendering to where the linked image will look for it.
   Open `tnt_candidates/<name>.czi` (or `.nd2`) in Icy. The image is not copied.
2. **`--sidecar`.** Writes the candidates as `<name>.xml` next to the original image. Use this only if you
   are sure no annotation file there needs to be kept. The script renames an existing file to a `.bak` copy first.

In Icy, with the ManualTNT plugin:
- Press **Space** to enter edit mode.
- Point at a candidate and press **Delete** to remove it. On a Mac keyboard that is **fn + Delete**.
- Click and drag to add a missed TNT.

## How it works

1. **Stack geometry.** A first plane that is a projection rather than a focal plane (this happens in the
   Nikon stacks) is detected and dropped. The substrate is the bottom 1 µm starting at the first plane that
   contains cells. Candidates are searched in the planes above the substrate.
2. **Cells.** Cellpose-SAM segments whole cells on the membrane channel (Cy5 for Nikon, AF647 for Zeiss),
   with DAPI as the second channel.
3. **Lines.** In each plane above the substrate, two detectors propose straight lines:
   - a white top-hat filter with a 5-noise-sigma threshold, outside the thick cell bodies;
   - the actin-fibre and filopodia classes of the FAST U-Net++ F-actin model, applied at native resolution.
   Both go through a probabilistic Hough transform (segments of at least 3 µm, gaps up to 1.5 µm).
4. **Bridge rule.** A line is a candidate when:
   - at least 50% of its middle (ends trimmed by 20%) lies outside every cell,
   - both ends come within 3 µm of two different cells,
   - the detector mask covers at least 70% of the line,
   - its z-profile peaks above the substrate.
5. **Ranking.** Each cell pair keeps its best line. Pairs are ranked by the mean FAST filopodia probability
   along the line. Filopodia rather than stress fibres are the signal, because TNTs look more like filopodia.

## Results

These numbers come from the notebook and from the detector evaluation on annotated data.

- **Nikon, 8 TNTs marked by an expert on the tubes** (wt002 and wt006): 6 of the 8 are in the top 20 of their
  stack, and 5 are in the top 10. Seven of the 8 appear in the list at all. The eighth is not in the list,
  because its tube runs over a cluster of cells and the bridge rule rejects that by design.
- **Zeiss, annotated cell pairs** (10 stacks, 51 distinct pairs): 18 of the 51 are in the top 20 per stack.
  These annotations mark cell centres, not the tube, so they show which cells were connected, not where the
  tube runs.
- **Dense cultures** (Eps8 stacks, 21 annotated stacks, 250 annotated pairs): the top 20 per stack contains
  20 of the 95 annotated pairs on the two-channel stacks that were not used for tuning, and 61 of 155 on the
  three-channel stacks. Each stack has hundreds of candidate pairs, so expect many false candidates.

What this means in practice: the top of the list is worth reviewing, and most annotated TNTs are not in
the top 20. Accuracy is not reported as a single number, because most cell pairs are not TNTs, and nobody
has labelled all the negatives.

## Requirements

- **Python environment** with: `numpy`, `pandas`, `scipy`, `scikit-image`, `scikit-learn`, `matplotlib`,
  `nd2`, `aicspylibczi`, `cellpose` (version 4 or later, for Cellpose-SAM), `torch` (with MPS, CUDA or CPU),
  `nbformat` and `nbclient` (to run the notebook).
- **FAST model weights.** `fast_model.pth` (about 104 MB) and `model.py` from the BioImage.IO model
  `reliable-waterbuffalo` (Aljapur et al., J Cell Sci 2026, doi:10.1242/jcs.264265, CC-BY-4.0). They are not
  in this repository. Put them in `experiments_v4/models/fast_actin/`.
- **Helper modules.** The script and the notebook import the helper modules listed below. Keep these
  files in the same folder layout.

| Module | Folder | Used for |
|---|---|---|
| `v4_utils.py` | `experiments_v4/` | Cellpose cells, FAST, bridge rule |
| `hough_v3_utils.py` | `experiments_hough_v3/` | Stack geometry, gap-noise lines, line support |
| `hough_v2_utils.py` | `experiments_hough_v2/` | Imported by the v3 utilities |
| `gt_utils.py`, `nd2_utils.py`, `tnt_pairs.py` | `experiments/` | Data loading, ground truth, pair utilities |

The folder names are the ones used by the code. The code finds the modules through paths relative to these
folders, so keep the layout.

**Data paths.** `run_tnt_detection_pipeline.py` looks for images in two folders next to the script
(`Rapamycin DsRed SH+GFP H2B-1/` and `Anna images-TNTs/`). The notebook and the helper modules point to the
original data folder by absolute path. Edit `ND2_DIR`, `DEFAULT_DIR` and `DATA_DIRS` to match your data.

**Icy.** Needed only for review. The ManualTNT plugin is an Icy plugin distributed by the Institut Pasteur
image analysis platform.

## Running the notebook

Open `v4_nd2_wt006.ipynb` with the `microscopy` kernel and run all cells. Cell 1 sets `DETECTOR`, `SCORE`
and the device. The notebook saves its figures to `experiments_v4/results/`.

## Limitations

- The detector finds lines between cell pairs. It does not trace tubes. The output is a list of places to
  check, not a segmentation.
- Dense cultures produce many false candidates, because adjacent cells touch and cell borders look like
  tubes in the actin channel.
- Settings were chosen on a small number of stacks. The 8 expert-marked TNTs were also used while looking at
  the results, so those numbers are not an independent test.
- Annotated TNTs in the Zeiss and Eps8 sets are marked between cell centres. They do not show the tube path.
- The FAST model is trained on other cell types. Its filopodia score is the most useful cue we found, but it
  is not specific to TNTs.

## Citation and credits

- Cellpose-SAM: Pachitariu, Rariden and Stringer, bioRxiv 2025.
- FAST: Aljapur, Gardner, Carayanniotis and Harris, J Cell Sci 2026, doi:10.1242/jcs.264265.
- Icy and the ManualTNT plugin: de Chaumont et al., Nature Methods 2012.
- Sato tubeness: Sato et al., Medical Image Analysis 1998.

Authors: Caglayan Tuna, Anna Pepe, François Laurent, Chiara Zurzolo, Jean-Baptiste Masson.

## License

[To be chosen by the authors.]
