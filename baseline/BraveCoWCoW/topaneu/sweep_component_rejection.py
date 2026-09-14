#!/usr/bin/env python3
"""P3 rule sweep over cached component features; no network inference."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

ROOT=Path('/data/cyf/shared_data/TopAneu/BraveCoWCoW_predroi_160')
GT_STRUCTURE=ndimage.generate_binary_structure(3,2)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=ROOT); ap.add_argument('--base-rules',nargs='+',default=['threshold_0.05_minvox_10','threshold_0.3_minvox_5','threshold_0.4_minvox_1']); ap.add_argument('--distances',nargs='+',type=float,default=[3,5]); ap.add_argument('--conf',nargs='+',type=float,default=[.2,.3]); ap.add_argument('--margin',nargs='+',type=float,default=[.02,.05]); ap.add_argument('--contact',nargs='+',type=int,default=[0]); args=ap.parse_args()
    evalroot=args.root/'stagec_pred_oof_v2/evaluations'; cache=args.root/'stagec_pred_oof_v2/probabilities'; out= args.root/'stagec_pred_oof_v2/rejection_evaluations'; out.mkdir(parents=True,exist_ok=True)
    rules=[(d,c,m,k) for d in args.distances for c in args.conf for m in args.margin for k in args.contact]
    for base in args.base_rules:
        candidate=evalroot/base
        if not candidate.is_dir(): raise FileNotFoundError(candidate)
        for d,c,m,k in rules:
            name=f'{base}__dist{d:g}mm_conf{c:g}_margin{m:g}_contact{k}'
            ruleout=out/name; rows=[]
            for fold in range(5):
                src=candidate/f'fold_{fold}/validation_location_masks_component'; dst=ruleout/f'fold_{fold}/validation_location_masks_component'; dst.mkdir(parents=True,exist_ok=True)
                table={}
                csvpath=candidate/'components.csv'
                if csvpath.exists():
                    with csvpath.open() as h:
                        for r in csv.DictReader(h): table[(r['case'],int(r['component_id']))]=r
                for p in sorted(src.glob('*.nii.gz')):
                    case=p.name[:-7]; pred=sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.uint8); z=np.load(next((cache/f'fold_{fold}').glob(case+'.npz'))); prob=z['aneurysm_probability']; cc,n=ndimage.label(prob>=float(base.split('_')[1]),GT_STRUCTURE); result=np.zeros_like(pred)
                    for cid in range(1,n+1):
                        comp=cc==cid; r=table.get((case,cid));
                        if r is None: continue
                        keep=(float(r['nearest_vessel_distance_mm'])<=d and float(r['stagec_top1_probability'])>=c and float(r['stagec_margin'])>=m and int(r['vessel_contact_voxels'])>=k)
                        if keep: result[comp]=pred[comp]
                    ref=sitk.ReadImage(str(p)); image=sitk.GetImageFromArray(result); image.CopyInformation(ref); sitk.WriteImage(image,str(dst/p.name),True)
            (ruleout/'rule.json').write_text(json.dumps({'base':base,'max_distance_mm':d,'min_confidence':c,'min_margin':m,'min_contact_voxels':k},indent=2))
    print(f'generated rejection rules: {len(list(out.iterdir()))}')
if __name__=='__main__': main()
