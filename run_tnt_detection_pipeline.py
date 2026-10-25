#!/users/ctuna/microscopy/bin/python
"""Detect TNT candidates in one z-stack and write them as an Icy ManualTNT annotation file.

    /users/ctuna/microscopy/bin/python run_tnt_detection_pipeline.py 1
    /users/ctuna/microscopy/bin/python run_tnt_detection_pipeline.py 400k-sh-wt002
    /users/ctuna/microscopy/bin/python run_tnt_detection_pipeline.py "/path/to/image.czi" --top 30
    /users/ctuna/microscopy/bin/python run_tnt_detection_pipeline.py "/path/to/folder"

The input can be an image name (`1`, `1.czi`, `400k-sh-wt002`) -- looked up in the two data
folders -- a path to a .czi/.nd2, or a FOLDER: every .czi and .nd2 directly inside it is processed
(one failing image is reported and skipped, the others still run).

OUTPUT (default: `<image folder>/tnt_candidates/`)
  <name>.xml               Icy file in the same format as the annotators' `N.xml`: the image metadata
                           and a `plugins.fab.manualtnt.ManualTNT` block holding one <cell> marker per
                           detected cell and one <tnt> line per candidate, best candidate first, in
                           full-resolution image pixels.
  <name>_candidates.csv    one row per candidate: rank, the two cells, the z plane where the line is
                           best seen, its end points and scores. Row order = <tnt> order in the xml.

USING IT IN ICY
  Icy reads the ManualTNT state from the xml that has the same name as the image and sits next to it
  (`1.xml` for `1.czi`). For the czi set that file already holds the annotators' TNTs, so by default
  this script does NOT write there. Instead it puts a LINK to the image in the output folder, next to
  the candidate xml (no copy of the stack), and copies the plugin's cached rendering where the linked
  image will look for it: open `tnt_candidates/<name>.czi` in Icy. Or pass --sidecar: the script then writes `<name>.xml` next to the image after renaming the existing
  one to `<name>.xml.bak-<timestamp>`. Open the image in Icy with the ManualTNT plugin, delete the
  candidates that are not TNTs, add missed ones, and fix the cell markers.

METHOD (= experiments_v4 notebooks, default settings)
  1. Stack geometry: an nd2 z0 that is a projection is dropped; substrate = bottom 1 um from the first
     plane with cells; candidates are searched in the planes above it.
  2. Cells: Cellpose-SAM on the membrane channel (nd2: Cy5/WGA; czi: AF647) + DAPI.
  3. Lines per plane on the tube channel (nd2: TRITC; czi: AF647): v3's 5-sigma gap-noise detector
     plus the FAST U-Net++ F-actin mask, both through a probabilistic Hough transform.
  4. A candidate bridges a gap: >= half of the line's middle lies outside every cell, its two ends
     reach two different cells (within 3 um), the detector covers >= 70% of it, and its z-profile
     peaks above the substrate.
  5. One candidate per cell pair, its best line; ranked by FAST's filopodia probability along it.
  A ranked candidate list for review, not a validated detector: on the 8 TNTs Anna marked
  (wt002, wt006) 5 rank in the top 10 of their image; on the czi annotation 18 of 51 annotated
  pairs are in the top 20 per image.

Requires the microscopy env (cellpose, torch, nd2, aicspylibczi) and the models in
experiments_v4/models/.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIRS = [ROOT / 'Rapamycin DsRed SH+GFP H2B-1', ROOT / 'Anna images-TNTs']
sys.path.insert(0, str(ROOT / 'experiments_v4'))
import v4_utils as v4          # noqa: E402  (also puts experiments, experiments_hough_v2/v3 on sys.path)
import hough_v3_utils as h3    # noqa: E402

ICY_BLOCK = 'plugins.fab.manualtnt.ManualTNT'

# fixed settings, as in the v4 notebooks
CZI_DS = 2                     # czi working grid: 0.0706 -> 0.141 um/px
K_SIGMA, COVERAGE_MIN = 5.0, 0.7
REACH_UM, FREE_MIN = 3.0, 0.5
CELL_UM = 30.0


# ------------------------------------------------------------------------------- input

def find_image(name: str) -> Path:
    p = Path(name).expanduser()
    if p.suffix.lower() in ('.czi', '.nd2') and p.exists():
        return p.resolve()
    stem = p.stem if p.suffix.lower() in ('.czi', '.nd2') else p.name
    hits = [d / f'{stem}{ext}' for d in DATA_DIRS for ext in ('.czi', '.nd2') if (d / f'{stem}{ext}').exists()]
    if not hits:
        sys.exit(f'image "{name}" not found (give a .czi/.nd2 path, or a name in {[str(d) for d in DATA_DIRS]})')
    return hits[0]


def _pick(channels, wanted, default):
    for i, c in enumerate(channels):
        if wanted.lower() in c.lower():
            return i
    return default


def load_stack(path: Path):
    """-> dict(vols={'dapi','tube','membrane'}, px_um, dz_um, ds, channels, kind)."""
    if path.suffix.lower() == '.nd2':
        import nd2
        with nd2.ND2File(str(path)) as f:
            arr = f.asarray()                                   # (Z, C, Y, X)
            chans = [c.channel.name for c in f.metadata.channels]
            vx = f.voxel_size()
        get = lambda i: arr[:, i].astype(np.float32)
        # Nikon sets: DAPI / TRITC (actin) / Cy5 (WGA). Eps8 sets: GFP (fill) / DAPI / Cy5 (actin), or DAPI / Cy5.
        # The tubes are on TRITC when present, otherwise on Cy5; the cells use Cy5 (or TRITC without Cy5).
        dapi = _pick(chans, 'DAPI', 0)
        has_trit, has_cy5 = any('TRITC' in c for c in chans), any('Cy5' in c for c in chans)
        tube_name = 'TRITC' if has_trit else 'Cy5'
        tube = _pick(chans, tube_name, len(chans) - 1)
        mem = _pick(chans, 'Cy5', tube) if has_cy5 else tube
        return dict(vols=dict(dapi=get(dapi), tube=get(tube), membrane=get(mem)),
                    px_um=float(vx.x), dz_um=float(vx.z), ds=1, channels=chans, kind='nd2',
                    tube_name=chans[tube], membrane_name=chans[mem])
    from aicspylibczi import CziFile
    cf = CziFile(str(path))
    meta = cf.meta
    chans = [c.get('Name') or (c.findtext('Fluor') or f'C{i}')
             for i, c in enumerate(meta.findall('.//Information/Image/Dimensions/Channels/Channel'))]
    scale = {d.get('Id'): float(d.findtext('Value')) * 1e6 for d in meta.findall('.//Scaling/Items/Distance')}
    Z = dict(cf.get_dims_shape()[0])['Z'][1]
    ds = CZI_DS

    def get(ci):
        return np.stack([np.squeeze(cf.read_image(C=ci, Z=z)[0])[::ds, ::ds] for z in range(Z)]).astype(np.float32)
    dapi, tube = _pick(chans, 'DAPI', 0), _pick(chans, '647', len(chans) - 1)
    vt = get(tube)
    return dict(vols=dict(dapi=get(dapi), tube=vt, membrane=vt),
                px_um=scale.get('X', 0.0706) * ds, dz_um=scale.get('Z', 0.35), ds=ds, channels=chans,
                kind='czi', tube_name=chans[tube], membrane_name=chans[tube])


# ---------------------------------------------------------------------------- detection

def detect(S: dict, device: str = 'mps', log=print):
    from skimage.morphology import dilation, disk
    from skimage.transform import probabilistic_hough_line
    import pandas as pd

    vols, px, dz = S['vols'], S['px_um'], S['dz_um']
    G = h3.stack_geometry(vols['tube'], dz, substrate_um=1.0)
    SUB, MID, LAYER = G['SUBSTRATE'], G['MID'], G['layer']
    log(f'  geometry: z0 {"projection, dropped" if G["is_projection"] else "real"}; '
        f'substrate {SUB}; searched z{MID[0]}-z{MID[-1]}')

    cells = v4.cells_from_membrane(vols['membrane'], vols['dapi'], MID, px, cell_um=CELL_UM)
    log(f'  cells: {cells.max()} (Cellpose on {S["membrane_name"]} + DAPI)')

    fast = v4.load_fast(device)
    sig = h3.sigma_volume(vols['tube'], LAYER, cells > 0)
    no_floor = np.zeros(cells.shape, bool)
    rows = []
    for z in MID:
        d = h3.gap_lines(vols['tube'][z], px, k_sigma=K_SIGMA)
        P = v4.fast_plane(fast, vols['tube'][z], device)
        bw = (P[1] + P[4]) > 0.5
        sources = [('gap', d['lines'], d['binary_dil']),
                   ('fast', probabilistic_hough_line(bw, threshold=5, line_length=int(3 / px),
                                                     line_gap=int(1.5 / px), rng=0), dilation(bw, disk(1)))]
        for src, segs, support in sources:
            for seg in segs:
                pair, info = v4.bridge_test(seg, cells, px, reach_um=REACH_UM)
                if pair is None or info['free_frac'] < FREE_MIN:
                    continue
                if h3.line_support(seg, support, d['fg'], px)['coverage'] < COVERAGE_MIN:
                    continue
                f = h3.line_features(seg, z, sig, LAYER, SUB, cells > 0, no_floor, px, dz)
                if f['z_peak'] in SUB:
                    continue
                (x0, y0), (x1, y1) = seg
                rows.append(dict(src=src, z=int(z), cell_a=min(pair), cell_b=max(pair),
                                 x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1),
                                 filopodia=v4.mean_along(P[4], seg), fibres=v4.mean_along(P[1], seg),
                                 contrast=h3.line_contrast(seg, d['tophat'], d['gap_median'], d['gap_sigma']),
                                 over_gap=info['free_frac'], length_um=f['length_um'], elev_um=f['elev_um']))
    lines = pd.DataFrame(rows)
    if len(lines) == 0:
        return lines, cells, G
    cand = (lines.sort_values('filopodia', ascending=False)
            .drop_duplicates(['cell_a', 'cell_b']).reset_index(drop=True))
    cand.insert(0, 'rank', np.arange(1, len(cand) + 1))
    return cand, cells, G


# ------------------------------------------------------------------------------- output

def write_icy_xml(out_xml: Path, image: Path, S: dict, cand, cells, template: Path | None):
    """Icy document with a ManualTNT block: cell markers + candidate lines, full-resolution pixels."""
    from skimage.measure import regionprops
    ds = S['ds']
    if template is not None and template.exists():
        root = ET.parse(template).getroot()                      # keep the image's metadata / LUT
        for old in root.findall(ICY_BLOCK):
            root.remove(old)
    else:
        root = ET.Element('root')
        ET.SubElement(root, 'name').text = image.stem
        meta = ET.SubElement(root, 'meta')
        for k, v in (('pixelSizeX', S['px_um'] / ds), ('pixelSizeY', S['px_um'] / ds), ('pixelSizeZ', S['dz_um'])):
            ET.SubElement(meta, k).text = repr(float(v))
        for i, c in enumerate(S['channels']):
            ET.SubElement(meta, f'channelName{i}').text = c
        ET.SubElement(root, 'rois')
        ET.SubElement(root, 'overlays')
    block = ET.SubElement(root, ICY_BLOCK)
    cell_el = ET.SubElement(block, 'cell')
    for r in regionprops(cells):
        cy, cx = r.centroid
        ET.SubElement(cell_el, 'cell', {'class': '0', 'x': repr(float(cx * ds)), 'y': repr(float(cy * ds))})
    tnt_el = ET.SubElement(block, 'tnt')
    for c in cand.itertuples():                                  # best candidate first
        ET.SubElement(tnt_el, 'tnt', {'x1': repr(c.x0 * ds), 'x2': repr(c.x1 * ds),
                                      'y1': repr(c.y0 * ds), 'y2': repr(c.y1 * ds)})
    ET.indent(root, space='')
    out_xml.write_text(ET.tostring(root, encoding='unicode') + '\n')


# --------------------------------------------------------------------------------- main

def prepare_for_icy(image: Path, out_dir: Path):
    """Make `out_dir` openable in Icy as is: a link to the image next to its candidate xml, and the
    ManualTNT plugin's cached rendering where the plugin will look for it.

    The plugin shows a denoised projection of every image it opens and caches it as
    `<image folder>/../cached rendering/<name>.render.tif`. For the linked image that is
    `<out_dir>/../cached rendering/`, a different folder from the original's, so without a copy the
    plugin recomputes the rendering of the whole stack -- which can exhaust Icy's memory (4 GB by
    default) and leave the image half-rendered.
    """
    link = out_dir / image.name
    if not link.exists() and image.parent.resolve() != out_dir.resolve():
        try:
            link.symlink_to(image)
            print(f'  linked {link.name} -> {image}')
        except OSError as e:
            print(f'  could not link the image ({e}); copy it next to the xml to open it in Icy')
    src = image.parent / '..' / 'cached rendering'
    dst = out_dir / '..' / 'cached rendering'
    if src.resolve() != dst.resolve():
        for ext in ('.render.tif', '.render.xml'):
            f = src / f'{image.name}{ext}'
            if f.exists() and not (dst / f.name).exists():
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, dst / f.name)
                print(f'  copied the plugin\'s cached rendering to {dst.resolve() / f.name}')


def process(image: Path, out_dir: Path, top: int | None, sidecar_flag: bool, device: str,
            link_image: bool = True):
    t0 = time.time()
    print(f'{image.name}')
    S = load_stack(image)
    print(f'  {S["kind"]}: {S["vols"]["tube"].shape} at {S["px_um"]:.4f} um/px (working grid), dz {S["dz_um"]} um; '
          f'tubes on {S["tube_name"]}')
    cand, cells, G = detect(S, device=device)
    if top:
        cand = cand.head(top)
    print(f'  {len(cand)} candidate TNTs ({time.time() - t0:.0f} s)')

    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar = image.with_suffix('.xml')
    out_xml = out_dir / f'{image.stem}.xml'
    write_icy_xml(out_xml, image, S, cand, cells, template=sidecar)
    csv = cand.copy()
    for k in ('x0', 'y0', 'x1', 'y1'):                          # full-resolution pixels, as in the xml
        if k in csv:
            csv[k] = csv[k] * S['ds']
    csv.round(3).to_csv(out_dir / f'{image.stem}_candidates.csv', index=False)
    print(f'  wrote {out_xml}')
    print(f'        {out_dir / (image.stem + "_candidates.csv")}')

    if link_image:
        prepare_for_icy(image, out_dir)

    if sidecar_flag:
        if sidecar.exists():
            backup = sidecar.with_name(f'{sidecar.name}.bak-{time.strftime("%Y%m%d-%H%M%S")}')
            shutil.move(sidecar, backup)
            print(f'  existing {sidecar.name} renamed to {backup.name}')
        shutil.copy(out_xml, sidecar)
        print(f'  wrote {sidecar}  (Icy will load it with the image)')
    return len(cand)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='image name (e.g. 1, 1.czi, 400k-sh-wt002), path to a .czi/.nd2, '
                                  'or a folder: every .czi/.nd2 in it is processed')
    ap.add_argument('--top', type=int, default=None, help='keep only the N best candidates per image (default: all)')
    ap.add_argument('--out', default=None, help='output folder (default: <image folder>/tnt_candidates)')
    ap.add_argument('--sidecar', action='store_true',
                    help='also write <name>.xml next to each image so Icy loads it with the image; '
                         'an existing one is first renamed to <name>.xml.bak-<timestamp>')
    ap.add_argument('--device', default='mps', help="torch device: 'mps' (default), 'cuda' or 'cpu'")
    ap.add_argument('--no-link', action='store_true',
                    help='do not put a link to each image in the output folder (by default the folder can '
                         'be opened in Icy directly: image link + candidate xml side by side)')
    a = ap.parse_args()

    src = Path(a.input).expanduser()
    if src.is_dir():
        images = sorted([p for p in src.iterdir() if p.suffix.lower() in ('.czi', '.nd2') and not p.name.startswith('.')],
                        key=lambda p: (p.suffix, int(p.stem) if p.stem.isdigit() else 10 ** 9, p.stem))
        if not images:
            sys.exit(f'no .czi or .nd2 files in {src}')
        print(f'{len(images)} images in {src}\n')
    else:
        images = [find_image(a.input)]

    done, failed = [], []
    for image in images:
        out_dir = Path(a.out).expanduser() if a.out else image.parent / 'tnt_candidates'
        try:
            done.append((image.name, process(image, out_dir, a.top, a.sidecar, a.device, not a.no_link)))
        except Exception as e:                                   # keep going with the other images
            failed.append((image.name, f'{type(e).__name__}: {e}'))
            print(f'  FAILED: {type(e).__name__}: {e}')
        print()
    if len(images) > 1:
        print(f'done: {len(done)} images, {sum(n for _, n in done)} candidates in total'
              + (f'; FAILED: {len(failed)}' if failed else ''))
        for name, err in failed:
            print(f'  {name}: {err}')


if __name__ == '__main__':
    main()
