# Project-page assets

- `teaser.png` and `pipeline.png`: exported from the final arXiv manuscript figures, at 2600 px and 2400 px widths respectively. The paper remains the source of truth.
- `paired-gains.svg`: generated from `results/paper_table1_local_pair_averages.json`; includes every one of the nine matched host pairs in paper order. Scores are displayed paper values and gains are percentage points.
- `dataset-overview.svg`: schematic annotation workflow, not a measured or fixed event sequence. It contains no source images or annotator identities.
- `social-preview.png`: 1280 × 640 px repository-sharing artwork.

Rebuild the vector assets with Node.js:

```bash
node docs/assets/build-page-assets.mjs
```

Pass an optional output path to generate the self-contained social-preview SVG, for example `node docs/assets/build-page-assets.mjs /tmp/evivit-social-preview.svg`. Rasterize it at its declared 1280 × 640 size to reproduce `social-preview.png`.

Raster derivatives use the SVG at its declared size. Social preview artwork must be uploaded separately in GitHub repository Settings → Social preview; committing the image alone does not configure that setting.
