import os
import glob
import csv
import numpy as np
import SimpleITK as sitk

img_dir = "/data/cyf/codes/TopAneu/nnUNet_data/nnUNet_raw/Dataset501_TopAneu/imagesTr"
out_csv = "/data/cyf/codes/TopAneu/topaneu_intensity_stats.csv"

files = sorted(glob.glob(os.path.join(img_dir, "*_0000.nii.gz")))

def get_modality(name):
    n = name.lower()

    # 根据 TopAneu 文件名判断
    if "_ct_" in n or "_cta_" in n:
        return "CTA"

    if "_mr_" in n or "_mra_" in n:
        return "MRA"

    return "UNKNOWN"

rows = []

for i, f in enumerate(files):
    img = sitk.ReadImage(f)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)

    vals = arr[np.isfinite(arr)]

    # 同时统计非零区域，避免大量 padding 0 干扰
    nz = vals[vals != 0]

    def stats(x):
        if x.size == 0:
            return [np.nan] * 11

        return [
            float(np.min(x)),
            float(np.percentile(x, 0.1)),
            float(np.percentile(x, 0.5)),
            float(np.percentile(x, 1)),
            float(np.percentile(x, 50)),
            float(np.percentile(x, 99)),
            float(np.percentile(x, 99.5)),
            float(np.percentile(x, 99.9)),
            float(np.max(x)),
            float(np.mean(x)),
            float(np.std(x)),
        ]

    s_all = stats(vals)
    s_nz = stats(nz)

    modality = get_modality(os.path.basename(f))

    rows.append([
        os.path.basename(f),
        modality,
        *arr.shape,
        *img.GetSpacing(),
        *s_all,
        *s_nz,
    ])

    print(
        f"[{i+1:03d}/{len(files)}] "
        f"{modality:3s} "
        f"{os.path.basename(f)} "
        f"min={s_all[0]:.2f} "
        f"p0.5={s_all[2]:.2f} "
        f"median={s_all[4]:.2f} "
        f"p99.5={s_all[6]:.2f} "
        f"max={s_all[8]:.2f}"
    )

header = [
    "file", "modality",
    "z", "y", "x",
    "spacing_x", "spacing_y", "spacing_z",

    "all_min", "all_p0.1", "all_p0.5", "all_p1",
    "all_median",
    "all_p99", "all_p99.5", "all_p99.9",
    "all_max", "all_mean", "all_std",

    "nz_min", "nz_p0.1", "nz_p0.5", "nz_p1",
    "nz_median",
    "nz_p99", "nz_p99.5", "nz_p99.9",
    "nz_max", "nz_mean", "nz_std",
]

with open(out_csv, "w", newline="") as fp:
    writer = csv.writer(fp)
    writer.writerow(header)
    writer.writerows(rows)

print("\nSaved:", out_csv)

# 汇总不同模态
for modality in ["CTA", "MRA", "UNKNOWN"]:
    r = [x for x in rows if x[1] == modality]

    if not r:
        continue

    print("\n==============================")
    print(modality, "cases:", len(r))
    print("==============================")

    # all_min 在 index 8
    # all_median index 12
    # all_p99.5 index 14
    # all_max index 16
    mins = np.array([x[8] for x in r])
    medians = np.array([x[12] for x in r])
    p995 = np.array([x[14] for x in r])
    maxs = np.array([x[16] for x in r])

    print("case min     median:", np.median(mins))
    print("case median  median:", np.median(medians))
    print("case p99.5   median:", np.median(p995))
    print("case max     median:", np.median(maxs))
