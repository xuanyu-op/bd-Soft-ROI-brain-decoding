import glob
import numpy as np
import torch
import webdataset as wds
from tqdm import tqdm


def _to_scalar_id(x):
    """Convert a tensor, ndarray, list, tuple, or scalar into a Python int."""
    if isinstance(x, torch.Tensor):
        return int(x.detach().cpu().item())
    if isinstance(x, np.ndarray):
        return int(x.reshape(-1)[0].item())
    if isinstance(x, (list, tuple)):
        return _to_scalar_id(x[0])
    return int(x)


def make_test_dataset(data_path, subj):
    """
    Build the WebDataset pipeline for the specified subject's test set.
    """
    test_url_pattern = f"{data_path}/webdataset_avg_split/test/test_subj0{subj}_*.tar"
    test_urls = glob.glob(test_url_pattern)
    if not test_urls:
        raise FileNotFoundError(f"No test shards found for pattern: {test_url_pattern}")

    print(f"Loading full test set from {len(test_urls)} shards for subject {subj}...")
    dataset = wds.WebDataset(test_urls, resampled=False, shardshuffle=False) \
        .decode("torch").rename(images="jpg;png", voxels='nsdgeneral.npy', coco="coco73k.npy") \
        .to_tuple("voxels", "images", "coco")
    return dataset


def build_light_cache_for_prompt_opt(data_path, subj, coco_id_to_captions):
    """
    Build a lightweight cache for prompt optimization. Only samples whose coco_id exists in the caption dictionary are cached.
    """
    test_data = make_test_dataset(data_path, subj)
    cached_features = []

    print("Creating lightweight cache for Prompt Optimization (avoids storing large objects)...")
    with torch.no_grad():
        for voxels, images_tensor, coco_ids in tqdm(test_data, desc="Caching raw validation data"):
            coco_id = str(_to_scalar_id(coco_ids))
            if coco_id not in coco_id_to_captions:
                continue

            # Store voxels as float16 NumPy arrays to reduce memory usage;
            # convert them back to tensors when needed
            cached_features.append({
                'voxels_fp16_numpy': (voxels.detach().cpu().to(torch.float16).numpy()
                                      if isinstance(voxels, torch.Tensor)
                                      else np.asarray(voxels, dtype=np.float16)),
                'image_tensor': images_tensor.squeeze(0).cpu(),
                'coco_id': coco_id
            })

    print(f"✅ Cached {len(cached_features)} valid samples from the full test set.")
    if len(cached_features) == 0:
        raise RuntimeError("No valid samples were cached. Check your data paths and subject ID.")

    return cached_features