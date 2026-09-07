import os
import glob
import numpy as np
import SimpleITK as sitk

base = "/data/cyf/codes/TopAneu/nnUNet_data/nnUNet_raw/Dataset501_TopAneu"

img_dir = os.path.join(base, "imagesTr")
seg_dir = os.path.join(base, "labelsTr")

cta_files = sorted(
    f for f in glob.glob(os.path.join(img_dir, "*_0000.nii.gz"))
    if "_ct_" in os.path.basename(f).lower()
)

all_fg = []
positive_cases = 0
negative_cases = 0

for i, img_file in enumerate(cta_files):
    name = os.path.basename(img_file)
    case_id = name.replace("_0000.nii.gz", "")
    seg_file = os.path.join(seg_dir, case_id + ".nii.gz")

    if not os.path.isfile(seg_file):
        print("Missing label:", seg_file)
        continue

    img = sitk.GetArrayFromImage(
        sitk.ReadImage(img_file)
    ).astype(np.float32)

    seg = sitk.GetArrayFromImage(
        sitk.ReadImage(seg_file)
    )

    mask = seg > 0

    if not np.any(mask):
        negative_cases += 1
        print(f"[{i+1:03d}/{len(cta_files)}] NEGATIVE {case_id}")
        continue

    positive_cases += 1

    values = img[mask]
    values = values[np.isfinite(values)]

    all_fg.append(values)

    print(
        f"[{i+1:03d}/{len(cta_files)}] "
        f"{case_id} "
        f"voxels={len(values)} "
        f"min={values.min():.2f} "
        f"median={np.median(values):.2f} "
        f"max={values.max():.2f}"
    )

all_fg = np.concatenate(all_fg)

print("\n======================================")
print("CTA foreground intensity statistics")
print("======================================")
print("CTA cases:", len(cta_files))
print("positive CTA cases:", positive_cases)
print("negative CTA cases:", negative_cases)
print("foreground voxels:", len(all_fg))

print("min:", float(np.min(all_fg)))
print("p0.5:", float(np.percentile(all_fg, 0.5)))
print("mean:", float(np.mean(all_fg)))
print("median:", float(np.median(all_fg)))
print("std:", float(np.std(all_fg)))
print("p99.5:", float(np.percentile(all_fg, 99.5)))
print("max:", float(np.max(all_fg)))
