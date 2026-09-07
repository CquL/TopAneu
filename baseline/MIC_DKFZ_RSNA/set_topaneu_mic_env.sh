#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Please source this file: source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

# Keep this fork isolated from editable packages in user site-packages.
export PYTHONNOUSERSITE=1
export nnUNet_raw=/data/cyf/shared_data/TopAneu/MIC_DKFZ_RSNA_data/nnUNet_raw
export nnUNet_preprocessed=/data/cyf/shared_data/TopAneu/MIC_DKFZ_RSNA_data/nnUNet_preprocessed
export nnUNet_results=/data/cyf/shared_data/TopAneu/MIC_DKFZ_RSNA_data/nnUNet_results

echo "TopAneu MIC-DKFZ paths configured for Dataset502_TopAneuMIC"
