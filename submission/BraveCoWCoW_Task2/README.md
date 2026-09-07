# BraveCoWCoW TopAneu Task 2 submission

This is an isolated Grand Challenge algorithm container. It accepts either the
`head-ct-angiography` or `head-mr-angiography` interface and writes one uint8
0..52 mask to `/output/images/aneurysm-segmentation/output.mha`.

Model files are deliberately excluded from the container and are mounted at
`/opt/ml/model`. Upload the generated `model.tar.gz` as the Algorithm Model.

For a fast T4 smoke test, run one fold:

```bash
TOPANEU_FOLDS=0 bash do_test_run.sh
```

For the default five-fold sequential ensemble:

```bash
bash do_test_run.sh
```
