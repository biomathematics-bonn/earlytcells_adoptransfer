# ============================================================
# scRNA-seq adoptive transfer – 02 Doublet detection (scDblFinder)
#
# Input : RDS_integrated_Seurat_SCE_scTfh23.RData (from 01_preprocessing_integration.Rmd)
# Output: DoubletScanner_scTfh23.csv (per-cell scDblFinder class / score),
#         read in by 03_clustering_TB_annotation.Rmd
#
# Run in a separate conda environment with scDblFinder (package version conflicts with
# the Seurat environment). Doublets are called per sample (samples = "orig.ident").
# Reference: https://bioconductor.org/packages/3.17/bioc/vignettes/scDblFinder/inst/doc/scDblFinder.html
# ============================================================

set.seed(123)
library(scDblFinder)
library(BiocParallel)

project <- "scTfh23"
rdata   <- "../RData/"

load(file = paste0(rdata, "RDS_integrated_Seurat_SCE_", project, ".RData"))   # -> tfh.sce

tfh.sce <- scDblFinder(tfh.sce, samples = "orig.ident", BPPARAM = MulticoreParam(3))

write.csv(tfh.sce@colData, file = paste0(rdata, "DoubletScanner_", project, ".csv"))
