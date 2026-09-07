"""Component-level TopAneu location assignment using vessel topology."""

import numpy as np
from scipy import ndimage

# Vessel IDs are those in topaneu_release/vessel_mapping.json. Values are
# candidate parent vessels for each TopAneu location (junctions have >1).
LOCATION_VESSEL_CANDIDATES = {
    1:[23], 2:[24], 3:[29], 4:[30], 5:[23,29], 6:[24,30], 7:[1], 8:[23,24,1],
    9:[27], 10:[28], 11:[1,27], 12:[1,28], 13:[25], 14:[26], 15:[1,25], 16:[1,26],
    17:[1,2,3], 18:[2], 19:[3], 20:[21], 21:[22], 22:[35], 23:[36], 24:[4,33],
    25:[6,34], 26:[4], 27:[6], 28:[4,8], 29:[6,9], 30:[4,31], 31:[6,32],
    32:[4], 33:[6], 34:[4,11], 35:[6,12], 36:[10], 37:[11], 38:[12], 39:[15],
    40:[15], 41:[13], 42:[14], 43:[15,16], 44:[15,16], 45:[5], 46:[7],
    47:[5,17], 48:[7,19], 49:[5,17], 50:[7,19], 51:[17,18], 52:[19,20],
}


def assign_components(aneurysm_mask, vessel_mask, location_probs=None,
                      min_component_size=3, dilation_radius=3):
    """Assign one location ID to every connected aneurysm component.

    ``vessel_mask`` is a 36-class integer array. ``location_probs`` is an
    optional 52-vector from the classification head and acts as a soft prior;
    vessel contact/topology remains the primary signal.
    """
    aneurysm_mask = np.asarray(aneurysm_mask) > 0
    vessel_mask = np.asarray(vessel_mask)
    components, n_components = ndimage.label(aneurysm_mask)
    output = np.zeros(aneurysm_mask.shape, dtype=np.uint8)
    assignments = []
    probs = None if location_probs is None else np.asarray(location_probs, dtype=float)
    if probs is not None and probs.size != 52:
        raise ValueError(f'location_probs must have 52 entries, got {probs.size}')

    structure = ndimage.generate_binary_structure(3, 1)
    if dilation_radius > 0:
        structure = ndimage.iterate_structure(structure, dilation_radius)

    for component_id in range(1, n_components + 1):
        component = components == component_id
        size = int(component.sum())
        if size < min_component_size:
            continue
        near_vessels = vessel_mask[ndimage.binary_dilation(component, structure=structure)]
        near_vessels = set(int(v) for v in np.unique(near_vessels) if 1 <= int(v) <= 36)
        distance_scores = {}
        component_distance = ndimage.distance_transform_edt(~component)
        for location_id, candidates in LOCATION_VESSEL_CANDIDATES.items():
            candidate_mask = np.isin(vessel_mask, candidates)
            contact = int(np.count_nonzero(candidate_mask & ndimage.binary_dilation(component, structure=structure)))
            if candidate_mask.any():
                distance = float(component_distance[candidate_mask].min())
            else:
                distance = float('inf')
            topology = (2.0 if set(candidates) & near_vessels else 0.0) + min(contact, 1000) / 1000.0
            distance_scores[location_id] = topology - min(distance, 20.0) / 20.0
        scores = np.asarray([distance_scores[i] for i in range(1, 53)], dtype=float)
        if probs is not None:
            scores += np.log(np.clip(probs, 1e-6, 1.0))
        location_id = int(np.argmax(scores) + 1)
        output[component] = location_id
        assignments.append({
            'component_id': component_id,
            'size_voxels': size,
            'near_vessel_ids': sorted(near_vessels),
            'location_id': location_id,
            'score': float(scores[location_id - 1]),
        })
    return output, assignments

