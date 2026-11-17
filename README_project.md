# earlyTcells_adopTransfer

Analysis code for the manuscript

**Spatiotemporal dissection of early Tfh and Th1 differentiation reveals a pre-diversified naive T cell pool**

## Repository structure

| Folder | Content | Figures |
|---|---|---|
| [`scrnaseq_gitlab/`](scrnaseq_gitlab/) | scRNA-seq time course after adoptive transfer (days 0–6): preprocessing, integration, T-cell annotation, module scores; GEX + TCR + hashtag data of CD62L/KLF2-sorted naive CD4 T cells (mouse); dual-TCR analysis of a public human CD4 T-cell reference | Fig. 2, 3a, 5d–f, 6a–d, 6g–i; Suppl. Fig. 2b, 5 |
| [`xenium_gitlab/`](xenium_gitlab/) | Xenium In Situ data (two slides): QC, cell-type and Tg-subtype annotation, T/B-border inference, positioning of transgenic T cells relative to the border | Fig. 4; Suppl. Fig. 4 |
| [`mathematical_modelling/`](mathematical_modelling/) | Mathematical model of early T-helper differentiation and proliferation (ODE model, parameter fitting) | Fig. 3c–e; Suppl. Fig. 3b; Suppl. Tables 2–3 |

Every folder has its own README with the order of the scripts, inputs / outputs and a table that maps each
figure panel to the script and output file that produce it. Output files of manuscript panels carry the panel as
prefix (e.g. `Fig4c_...pdf`).

Flow-cytometry data (Fig. 1, 3b, 5b–c, 6e–f, 7; Suppl. Fig. 1, 3a, 6) were analysed with FlowJo / GraphPad Prism
and are not part of this repository.

## Data availability

| Data set | Repository | Accession |
|---|---|---|
| scRNA-seq time course (GEX + ADT) | zenodo |
| GEX + TCR + hashtags, CD62L/KLF2-sorted naive CD4 T cells | GEO | GSE354524 |
| Xenium In Situ (two slides, raw and processed) | GEO | GSE354577 |
| Human CD4 T-cell reference (Terekhova et al.) | Synapse | syn49637038 |

Paths in the scripts are relative to the respective folder; the expected layout of the data folders is
described in each folder's README.


