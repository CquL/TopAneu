# BraveCoWCoW multi-stage TopAneu Task 2 submission

This container runs the TopAneu multi-stage model: publisher Stage 1 ROI,
Dataset505 vessel-only segmentation, binary aneurysm segmentation, and the
Stage-C 52-location classifier. It accepts either the `head-ct-angiography`
or `head-mr-angiography` interface and writes one uint8 0..52 mask to
`/output/images/aneurysm-segmentation/output.mha`.

Model files are deliberately excluded from the container and are mounted at
`/opt/ml/model`. Upload the generated `model.tar.gz` as the Algorithm Model.

The default inference uses all five folds. It uses V2 fold 0 aneurysm weights,
the completed AneurysmOnly final weights for folds 1-4, VesselOnly best
weights, and Stage-C best weights. The default aneurysm probability threshold
is 0.10; override it with `TOPANEU_ANEURYSM_THRESHOLD` during local testing.

For a fast T4 smoke test, run one fold:

```bash
TOPANEU_FOLDS=0 bash do_test_run.sh
```

For the default five-fold sequential ensemble:

```bash
bash do_test_run.sh
```
