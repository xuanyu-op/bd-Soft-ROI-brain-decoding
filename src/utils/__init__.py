def safe_get_source(sources):
    # Check whether all source labels in the current batch are the same
    assert len(set(sources)) == 1
    # Return the unique source label
    return sources[0]


def adapt_voxels(voxels, training=False):
    # During training: if the input contains a repeat dimension, randomly select one repeat
    if training:
        if voxels.dim() == 3 or voxels.dim() == 5:
            repeat_index = random.randint(0, 2)
            voxels = voxels[:, repeat_index]
    # During inference/validation: if the input contains a repeat dimension, average over repeats
    else:
        if voxels.dim() == 3 or voxels.dim() == 5:
            voxels = voxels.mean(dim=1)

    # Return the processed voxel data
    return voxels