"""Cache full-ROI probabilities and scan binary thresholds without training."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy import ndimage


def counts(prob, target, threshold, minimum):
    structure = ndimage.generate_binary_structure(3, 2)
    labels, n = ndimage.label(prob > threshold, structure)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum
    keep[0] = False
    pred = keep[labels]
    gt_cc, gt_n = ndimage.label(target, structure)
    hits = np.unique(gt_cc[pred & target])
    hit_pred = np.unique(labels[pred & target])
    pred_n = int(keep[1:].sum())
    tp = int(np.count_nonzero(pred & target))
    fp = int(pred.sum()) - tp
    fn = int(target.sum()) - tp
    return tp, fp, fn, gt_n, len(hits), pred_n, pred_n - len(hit_pred)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--folds', nargs='+', type=int, default=list(range(5)))
    p.add_argument('--checkpoints', nargs='+', choices=['best', 'final'], default=['best', 'final'])
    p.add_argument('--limit', type=int, default=0, help='Smoke-test cases per fold; use a separate output')
    args = p.parse_args()
    from nnxnet.training.nnXNetTrainer.nnXNetTrainer_TopAneuAneurysmOnly import nnXNetTrainer_TopAneuAneurysmOnly as Trainer
    dataset = 'Dataset505_TopAneuBraveCoWCoWPredROI'
    plan = 'TopAneuBraveCoWCoWPredROIPlansV2_160'
    pre = Path(os.environ['nnXNet_preprocessed']) / dataset
    results = Path(os.environ['nnXNet_results']) / dataset / f'nnXNetTrainer_TopAneuAneurysmOnly__{plan}__3d_fullres'
    raw = Path(os.environ['nnXNet_raw']) / dataset
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    configs = [(t, n) for t in [.10, .15, .20, .25, .30, .40, .50] for n in [0, 5, 10, 20, 30]]
    for fold in args.folds:
        # Trainer construction also creates logs: redirect its result root to
        # the scan output so old experiment artifacts remain untouched.
        import nnxnet.training.nnXNetTrainer.nnXNetTrainer as base_module
        old_results = base_module.nnXNet_results
        base_module.nnXNet_results = str(out / 'runtime')
        try:
            trainer = Trainer(json.loads((pre / f'{plan}.json').read_text()), '3d_fullres', fold,
                              json.loads((pre / 'dataset.json').read_text()), device=torch.device('cuda'))
        finally:
            base_module.nnXNet_results = old_results
        trainer.initialize()
        keys = json.loads((pre / 'splits_final.json').read_text())[fold]['val']
        if args.limit:
            keys = keys[:args.limit]
        data_set = trainer.get_tr_and_val_datasets()[1]
        for kind in args.checkpoints:
            checkpoint = results / f'fold_{fold}' / f'checkpoint_{kind}.pth'
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            trainer.network.load_state_dict(state['network_weights'], strict=True)
            epoch = state['current_epoch']
            del state
            trainer.network.eval()
            trainer.set_deep_supervision_enabled(False)
            dest = out / f'fold_{fold}' / kind
            dest.mkdir(parents=True, exist_ok=True)
            identity = {'checkpoint': str(checkpoint), 'bytes': checkpoint.stat().st_size,
                        'mtime_ns': checkpoint.stat().st_mtime_ns, 'epoch': epoch,
                        'space': 'Dataset505 resized ROI', 'limit': args.limit}
            manifest = dest / 'manifest.json'
            if manifest.exists() and json.loads(manifest.read_text()) != identity:
                raise RuntimeError(f'Cache provenance differs: {manifest}; choose a new output')
            manifest.write_text(json.dumps(identity, indent=2))
            totals = {c: np.zeros(7, dtype=np.int64) for c in configs}
            with (dest / 'per_case.csv').open('w') as handle:
                writer = csv.writer(handle)
                writer.writerow(['case', 'threshold', 'min_voxels', 'tp', 'fp', 'fn', 'gt_lesions', 'detected_lesions', 'pred_components', 'fp_components'])
                for index, case in enumerate(keys):
                    cache = dest / f'{case}.npz'
                    if cache.exists():
                        with np.load(cache) as z:
                            prob = z['probabilities'].astype(np.float32)
                    else:
                        data, _, properties = data_set.load_case(case)
                        spatial = np.asarray(data.shape[1:])
                        patch = np.asarray(trainer.configuration_manager.patch_size)
                        if np.any(spatial > patch):
                            raise ValueError(f'{case}: ROI exceeds patch')
                        before = (patch - spatial) // 2
                        after = patch - spatial - before
                        pad = [int(v) for pair in zip(before[::-1], after[::-1]) for v in pair]
                        tensor = F.pad(torch.from_numpy(data[None]).cuda(), pad)
                        with torch.inference_mode(), torch.autocast('cuda'):
                            logits = trainer.network(tensor, aneurysm_only=True)
                        cropped = logits.float().softmax(1)[0, 1].detach().cpu().numpy()
                        cropped = cropped[tuple(slice(int(b), int(b+s)) for b, s in zip(before, spatial))]
                        prob = np.zeros(properties['shape_before_cropping'], dtype=np.float32)
                        prob[tuple(slice(a,b) for a,b in properties['bbox_used_for_cropping'])] = cropped
                        # Float32 cache preserves threshold comparisons on reruns.
                        temp = cache.with_suffix('.partial.npz')
                        np.savez_compressed(temp, probabilities=prob)
                        temp.replace(cache)
                        del tensor, logits
                    target = sitk.GetArrayFromImage(sitk.ReadImage(str(raw / 'labelsTr' / f'{case}.nii.gz'))) >= 37
                    if prob.shape != target.shape:
                        raise ValueError(f'Shape mismatch: {case}')
                    for cfg in configs:
                        values = counts(prob, target, *cfg)
                        totals[cfg] += values
                        writer.writerow([case, *cfg, *values])
                    handle.flush()
                    print(f'fold={fold} checkpoint={kind} epoch={epoch} cases={index+1}/{len(keys)}', flush=True)
            with (dest / 'summary.csv').open('w') as handle:
                writer = csv.writer(handle)
                writer.writerow(['threshold','min_voxels','dice','precision','recall','lesion_recall','fp_components','cases'])
                for cfg, (tp,fp,fn,gt,hits,pred_n,fp_n) in totals.items():
                    writer.writerow([*cfg,2*tp/max(2*tp+fp+fn,1),tp/max(tp+fp,1),tp/max(tp+fn,1),hits/max(gt,1),fp_n,len(keys)])
        del trainer
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
