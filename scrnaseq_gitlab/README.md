# scRNA-seq / TCR – adoptive transfer (Tfh / Th1) and naive CD4 T cells

Analysis code for
* the scRNA-seq time course of antigen-specific T cells after adoptive transfer and subcutaneous immunisation
  (draining lymph nodes, days 0, 2, 4, 6) – scripts 01–04,
* the GEX + TCR + hashtag data set of CD62L/KLF2-sorted naive CD4 T cells (mouse) – notebook 05,
* the dual-TCR analysis of a public human CD4 T-cell reference (Terekhova et al.) – notebook 06.

Data: see the data availability statement of the manuscript.

## Order of the scripts

| # | Script | Input | Output |
|---|---|---|---|
| 1 | `01_preprocessing_integration.Rmd` | Cell Ranger count matrices (GEX + ADT) | `RDS_initial_Seurat_scTfh23.rds`, `RDS_integrated_Seurat_scTfh23.rds`, `RDS_integrated_Seurat_SCE_scTfh23.RData` |
| 2 | `02_doublet_detection.R` | `RDS_integrated_Seurat_SCE_scTfh23.RData` | `DoubletScanner_scTfh23.csv` |
| 3 | `03_clustering_TB_annotation.Rmd` | integrated object, doublet calls, `DF_Manual Annotation.csv` | `RDS_integrated_Seurat_T_B.rds` |
| 4 | `04_Tcell_annotation_analysis.Rmd` | `RDS_integrated_Seurat_T_B.rds` | `RDS_Tcells_annotated.rds`, Fig. 2, 3a, 6d (right), Suppl. Fig. 2b, signature / ORA tables |
| 5 | `05_TCR_CD62L_KLF2_mouse.ipynb` (R kernel) | Cell Ranger multi output `SM_EBA_multi`, annotated T cells from 4 | Fig. 5d–f, 6a–d, Suppl. Fig. 5a |
| 6 | `06_human_naive_CD4_dualTCR.ipynb` (R kernel) | Terekhova et al. processed CD4 data + TCR contigs | Fig. 6g–i, Suppl. Fig. 5b |

All paths are relative to the script folder and are set in the `paths` chunk (01–04) or the second code cell (05, 06).
Expected layout (adapt the paths if your data live elsewhere):

```
project/
├── scrnaseq_gitlab/          # this repository (run the scripts from here)
├── data/
│   ├── 01_Counts/<Day>/      # Cell Ranger count output per sampling day (01)
│   ├── SM_EBA_multi/outs/    # Cell Ranger multi output, GEX + TCR + hashtags (05)
│   └── Terekhova_human_CD4/  # Synapse download: cd4_t_cells_extracted/cd4_t_cells/ + TCR contig files (06)
├── RData/                    # intermediate objects (01–05)
└── results/                  # DF_Manual Annotation.csv (03), figures/ and tables/
```
In 01, `recalculate_switch` / `recalculate_integration` = 1 recompute the objects from scratch; 0 loads the saved ones.

## Figure panels → code

Figure files carry the panel as prefix (e.g. `Fig2a_UMAP_celltypes.pdf`). Files starting with `F_` or without a
panel prefix are supporting plots that are not in the manuscript.

| Panel | Content | Script | Output file |
|---|---|---|---|
| Fig. 2a | UMAP, `Celltype_2` | 04 | `Fig2a_UMAP_celltypes.pdf` |
| Fig. 2b | Marker dot plot per population (`Celltype_3`, Th0 merged) | 04 | `Fig2b_DotPlot_markers_celltypes.pdf` |
| Fig. 2c | UMAP per day (downsampled to the smallest sample) | 04 | `Fig2c_UMAP_celltypes_per_day_downsampled.pdf` |
| Fig. 2d | Relative abundance per day | 04 | `Fig2d_StackedBarplot_celltypes_per_day.pdf` |
| Fig. 2e | (Pre-)Tfh / (pre-)Th1 genes, z-score heatmap | 04 | `Fig2e_Heatmap_TfhTh1_genes_zscore.pdf` |
| Fig. 3a | Module scores (z-standardised) per grouped population | 04 | `Fig3a_Tcell_module_scores_grouped_zscore.pdf` |
| Fig. 5a | Heatmap of Th0 / Pre-Tfh genes (quiescence, memory-like, …) | **not yet in the repository** | – |
| Fig. 5d | Reference UMAP, transferred `Celltype_2` | 05 | `Fig5d_refUMAP_predicted_celltypes.pdf` |
| Fig. 5e | Reference UMAP, CD62L/KLF2 population | 05 | `Fig5e_refUMAP_CD62L_KLF2_populations.pdf` |
| Fig. 5f | Population proportions per predicted cell type | 05 | `Fig5f_celltypes_by_population_proportion.pdf` |
| Fig. 6a | Fraction of dual-TCR cells per population | 05 | `Fig6a_dualTCR_fraction_per_population.pdf` |
| Fig. 6b | Clone sizes per population | 05 | `Fig6b_clone_sizes_per_population.pdf` |
| Fig. 6c | Dual-TCR cells per predicted cell type and population | 05 | `Fig6c_dualTCR_celltypes_by_population_counts.pdf` |
| Fig. 6d, left | Markers in CD62L⁻ populations, single vs dual TCR | 05 | `Fig6d_left_DotPlot_CD62Lneg_single_vs_dualTCR.pdf` |
| Fig. 6d, right | Markers in day-6 Tfh vs Th1 | 04 | `Fig6d_right_DotPlot_day6_Tfh_vs_Th1.pdf` |
| Fig. 6g | Lineage markers per cluster (human) | 06 | `Fig6g_DotPlot_lineage_markers_naive_clusters.pdf` |
| Fig. 6h | UMAP with annotated clusters (human) | 06 | `Fig6h_UMAP_annotated_clusters.pdf` |
| Fig. 6i | Dual-TCR frequency per donor and cluster (human) | 06 | `Fig6i_dualTCR_frequency_per_donor_boxplot.pdf` |
| Suppl. Fig. 2a | Heatmap of the top markers per `Celltype_2` | **not yet in the repository** | – |
| Suppl. Fig. 2b | Marker dot plots per day | 04 | `SuppFig2b_DotPlot_markers_by_day.pdf` |
| Suppl. Fig. 5a | Top genes of dual- vs single-TCR cells per population | 05 | `SuppFig5a_DotPlot_dualTCR_DEGs_per_population.pdf` |
| Suppl. Fig. 5b | UMAP coloured by Seurat cluster (human) | 06 | `SuppFig5b_UMAP_naive_clusters.pdf` |

Flow cytometry (Fig. 1, 3b, 5b–c, 6e–f, 7, Suppl. Fig. 1, 3a, 6) and the mathematical model (Fig. 3c–e,
Suppl. Fig. 3b, Suppl. Tables 2–3) are not part of this repository. Xenium panels (Fig. 4, Suppl. Fig. 4): see the
Xenium repository.

## What each step does

1. **Preprocessing** – genes in ≥ 3 cells, cells with ≥ 200 genes; QC filter: > 800 genes, < 10 % mitochondrial,
   < 0.05 % haemoglobin reads; per-sample log-normalisation and anchor-based integration (Seurat).
2. **Doublets** – scDblFinder per sample (separate conda environment).
3. **Clustering** – PCA (30 PCs), Louvain (resolution 1), UMAP; manual assignment of clusters to T and B cells.
4. **T cells** – embedding of all days (20 PCs, resolution 0.8), doublet removal, final annotation `Celltype_2`
   (Th0 I–VI, Pre-Tfh I/II/proliferating, Tfh, Pre-Th1 I/II/proliferating, Th1); composition per day
   (all cells and downsampled to the smallest sample), marker dot plots and heatmap, module scores
   (MSigDB HALLMARK / REACTOME / GO and custom signatures), one-sided Fisher over-representation of
   Day-6 Tfh / Th1 and Day-0 signatures in the markers of each population (BH-adjusted).
5. **CD62L/KLF2 populations (mouse)** – GEX QC (300–7,500 genes, < 10 % mitochondrial), hashtag demultiplexing
   (`HTODemux`, singlets), TCR clonotypes (scRepertoire, multi-chain cells kept; dual TCR = > 1 TRA or > 1 TRB chain),
   projection onto the reference UMAP of step 4 and label transfer of `Celltype_2` (`FindTransferAnchors` / `MapQuery`).
6. **Human reference** – author-normalised RNA and CITE-seq matrices (not re-normalised), CITE-seq gate for
   phenotypically naive CD4 T cells (CD20⁻ CD3⁺ CD4⁺ CD45RA⁺ CD45RO⁻ CD62L⁺), re-clustering, productive TRA/TRB chains
   counted per cell from the contig tables, dual-TCR frequency per donor and cluster (paired Wilcoxon, BH).

## Environment

01–04: R with Seurat v4 (conda environments *Seurat_toolbox2* / *Seurat_V4*), `dplyr`, `tidyverse`, `ggplot2`,
`patchwork`, `cowplot`, `msigdbr`, `DT`; `scDblFinder` and `BiocParallel` for step 2.
05–06: R 4.3 with Seurat v5 (Jupyter kernel *Seurat V5*), `scRepertoire`, `tidyverse`, `patchwork`,
`emmeans`; `umap-learn` (Python, via `reticulate`) for the reference UMAP model in 05.
