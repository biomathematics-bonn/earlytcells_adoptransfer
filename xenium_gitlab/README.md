# Xenium analysis – adoptive transfer (Tfh / Th1), draining lymph nodes

Analysis code for the Xenium In Situ data (two runs / slides, mouse draining lymph nodes, day 0–6 after immunisation).
Raw and processed data: GEO (Xenium series, see manuscript).

## Order of the notebooks

| # | Notebook | Input | Output |
|---|---|---|---|
| 1 | `Xenium_Run1_01_QC.ipynb` | Xenium Ranger re-segmentation, slide 0028164 (`test_segmentation4_0/outs`) | `20251009_Run1_reSeg_QC.h5ad` |
| 2 | `Xenium_Run2_01_QC.ipynb` | Xenium Ranger re-segmentation, slide 0028176 (`test_segmentation4_0_run2/outs`) | `20251009_Run2_reSeg_QC.h5ad` |
| 3 | `Xenium_Run1_02_annotation.ipynb` | `20251009_Run1_reSeg_QC.h5ad` | `ALLcells_run1.h5ad`, `20260417_TBcells_subannotated.h5ad` |
| 4 | `Xenium_Run2_02_annotation.ipynb` | `20251009_Run2_reSeg_QC.h5ad` | `20261019_ALLcells_run2.h5ad`, `20260520_TBcells_run2.h5ad` |
| 5 | `Xenium_03_TB_border_combined.ipynb` | the four files from 3 and 4 | Fig. 4b–g, Suppl. Fig. 4a/b (PDF/PNG) and per-section tables (CSV) |
| – | `schematics/SuppFig4c_make_qc_segments.py` | – | Suppl. Fig. 4c (scheme; needs `cairosvg`) |

All paths are relative (`BASE_DIR = ".."`) and are set in the second code cell of every notebook.
Expected layout (adapt `BASE_DIR` if your data live elsewhere):

```
project/
├── xenium_gitlab/            # this repository (run the notebooks from here)
├── 01_Raw/                   # Xenium Ranger re-segmentation output of both slides
│   ├── test_segmentation4_0/outs/
│   └── test_segmentation4_0_run2/outs/
└── 02_ProcessedData/         # h5ad files written by notebooks 01 and 02
```

Figures and tables of notebook 03 are written to the working directory.

## What each step does

**01 QC** – cell filter (area 20–80 µm², ≥ 80 transcripts, ≥ 10 genes, ≤ 99.5th percentile), gene filter (≥ 100 cells),
normalisation / log1p, assignment of every cell to its lymph-node section (`timepoint` = `<section>:Day<d>_<mouse>`),
first major cell-type annotation by module scores.

**02 Annotation** – major cell types by module score with a per-type threshold; Tg T-cell QC (MAD, Tukey, 2D doublets)
and top-25 % Tg-score filter; two-step Tg-subtype annotation (lineage Tfh vs Th1, then pre vs mature);
refined annotation of all cells (Tregs, cDC1/2, GC B cells, neutrophils, …); export.

**03 Combined** – cell-type composition per day (incl. FDCs), spatial maps and marker QC, T/B-border inference
with segment QC and parameter scan, Tg positioning relative to the border (fraction at the border, density in
border vs deep T / deep B zone, per-section ECDF of the signed distance).

Excluded section (QC): `6:Day3_0388`.

## Figure panels → code

Output files of the manuscript panels carry the panel as prefix (e.g. `Fig4c_...pdf`); all other plots written by
the notebooks are QC / supporting plots that are not in the manuscript.

| Panel | Content | Notebook (section) | Output file |
|---|---|---|---|
| Fig. 4a | Workflow scheme | – (drawn) | – |
| Fig. 4b, top | Cell-type annotation of one section | `Xenium_03` (3) | `SuppFig4a_Fig4b_<run>_spatial_celltypes.pdf/.png` |
| Fig. 4b, bottom | T zone / B zone with inferred T–B border | `Xenium_03` (4) | `Fig4b_TB_border_<run>_<section>_px20_s1.5_h75.png` |
| Fig. 4c | Mean cell-type fractions per day (incl. FDCs) | `Xenium_03` (2.2) | `Fig4c_Mean_cell_fractions_per_day_FDC.pdf` |
| Fig. 4d | Dot plot of the annotation markers | `Xenium_03` (3) | `Fig4d_Dotplot_annotation_markers_<section>.pdf` |
| Fig. 4e | Tg cells at the T–B border (% of all cells of the subset) | `Xenium_03` (5) | `Fig4e_Tg_fraction_at_TB_border_per_subset_bars.pdf` |
| Fig. 4f | ECDF of the distance to the border, per section mean ± SEM | `Xenium_03` (7) | `Fig4f_Tg_ECDF_per_LN_mean_sem.pdf` (+ `Fig4f_ECDF_per_LN_summary.csv`) |
| Fig. 4g | Median position relative to the border per day | `Xenium_03` (7), `METRIC = "median"` | `Fig4g_Tg_border_position_median_per_LN.pdf` (+ CSV) |
| Fig. 4h | Gene heatmap across the T–B border | **not yet in the repository** | – |
| Fig. 4i | Expression profiles across the T–B border per day | **not yet in the repository** | – |
| Suppl. Fig. 4a | Cell-type annotation of all sections | `Xenium_03` (3) | `SuppFig4a_Fig4b_<run>_spatial_celltypes.pdf/.png` |
| Suppl. Fig. 4b | UMAP coloured by cell type | `Xenium_03` (3) | `LN_umap_heatmaps/SuppFig4b_UMAP_celltypes_<section>.pdf` |
| Suppl. Fig. 4c | Segment-QC criteria (scheme) | `schematics/SuppFig4c_make_qc_segments.py` | `SuppFig4c_QC_segment_exclusion_criteria.svg/.pdf/.png` |
| Suppl. Table 4/5 | Xenium custom panel / marker genes | marker lists in the 02 notebooks | – |

## Environment

Python 3 with `scanpy`, `squidpy`, `anndata`, `spatialdata-io`, `numpy`, `pandas`, `scipy`, `scikit-image`,
`matplotlib`, `seaborn`.
