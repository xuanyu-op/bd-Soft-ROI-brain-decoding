import os
import random
import math
import json
import requests
import braceexpand
import numpy as np

import webdataset as wds
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# WebDataset node splitting function; return the input unchanged
def my_split_by_node(urls):
    return urls


# Return the corresponding loss function by name
def get_loss_func(recon_loss):
    loss_functions = {
        'mse': F.mse_loss,
        'l1': F.l1_loss,
        'huber': F.smooth_l1_loss,
        'quantile': lambda x, y: torch.quantile(torch.abs(x - y), 0.9)
    }
    if recon_loss not in loss_functions:
        raise ValueError(f"Unrecognized loss type: {recon_loss}")
    return loss_functions[recon_loss]


# Check whether the loss contains NaN; raise an error immediately if it does
def check_loss(loss):
    if loss.isnan().any():
        raise ValueError('NaN loss')


# Set random seeds for Python, NumPy, and PyTorch
def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        print('Note: not using cudnn.deterministic')


# Count and print the total number of model parameters and trainable parameters
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('param counts:\n{:,} total\n{:,} trainable'.format(total, trainable))


# Build the NSD WebDataset URLs on Hugging Face for the specified subject
def get_huggingface_urls(commit='main', subj=1):
    base_url = "https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/"
    train_url = base_url + commit + f"/webdataset_avg_split/train/train_subj0{subj}_" + "{0..17}.tar"
    val_url = base_url + commit + f"/webdataset_avg_split/val/val_subj0{subj}_0.tar"
    test_url = base_url + commit + f"/webdataset_avg_split/test/test_subj0{subj}_" + "{0..1}.tar"
    return train_url, val_url, test_url


# Build the WebDataset DataLoaders for the training set and validation set
def get_dataloaders(
        batch_size,
        image_var='images',
        num_devices=None,
        num_workers=None,
        train_url=None,
        val_url=None,
        meta_url=None,
        num_train=None,
        num_val=None,
        cache_dir="/tmp/wds-cache",
        voxels_key="nsdgeneral.npy",
        val_batch_size=None,
        to_tuple=["voxels", "images", "trial"],
        subj=1,
        data_ratio=1.0
):
    print("Getting dataloaders...")
    assert image_var == 'images'

    train_url = list(braceexpand.braceexpand(train_url))
    val_url = list(braceexpand.braceexpand(val_url))

    # Automatically download training data from Hugging Face if it does not exist locally
    if not os.path.exists(train_url[0]):
        print("downloading NSD from huggingface...")
        os.makedirs(cache_dir, exist_ok=True)

        train_url, val_url, test_url = get_huggingface_urls("main", subj)
        train_url = list(braceexpand.braceexpand(train_url))
        val_url = list(braceexpand.braceexpand(val_url))
        test_url = list(braceexpand.braceexpand(test_url))

        from tqdm import tqdm

        for url in tqdm(train_url):
            destination = cache_dir + "/webdataset_avg_split/train/" + url.rsplit('/', 1)[-1]
            print(f"\nDownloading {url} to {destination}...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(destination, 'wb') as file:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            file.write(chunk)

        for url in tqdm(val_url):
            destination = cache_dir + "/webdataset_avg_split/val/" + url.rsplit('/', 1)[-1]
            print(f"\nDownloading {url} to {destination}...")
            response = requests.get(url)
            response.raise_for_status()
            with open(destination, 'wb') as file:
                file.write(response.content)

        for url in tqdm(test_url):
            destination = cache_dir + "/webdataset_avg_split/test/" + url.rsplit('/', 1)[-1]
            print(f"\nDownloading {url} to {destination}...")
            response = requests.get(url)
            response.raise_for_status()
            with open(destination, 'wb') as file:
                file.write(response.content)

    # Automatically infer the number of devices and workers
    if num_devices is None:
        num_devices = torch.cuda.device_count()

    if num_workers is None:
        num_workers = int(os.getenv("NUM_WORKERS", str(num_devices)))
    try:
        num_workers = int(num_workers)
    except Exception:
        num_workers = 0
    num_workers = max(0, num_workers)

    # If the sample counts are not explicitly specified, read them from the metadata
    if num_train is None:
        metadata = json.load(open(meta_url))
        num_train = metadata['totals']['train']
    if num_val is None:
        metadata = json.load(open(meta_url))
        num_val = metadata['totals']['val']

    if val_batch_size is None:
        val_batch_size = batch_size

    # Compute batch information for the training stage
    global_batch_size = max(1, batch_size * max(1, num_devices))
    num_batches = max(1, math.floor(num_train / global_batch_size))
    num_worker_batches = num_batches if num_workers <= 1 else max(1, math.floor(num_batches / num_workers))

    print("\nnum_train", num_train)
    print("global_batch_size", global_batch_size)
    print("batch_size", batch_size)
    print("num_workers", num_workers)
    print("num_batches", num_batches)
    print("num_worker_batches", num_worker_batches)

    # Build the training WebDataset pipeline
    num_samples = int(num_train * data_ratio)
    train_data = wds.WebDataset(train_url, resampled=True, cache_dir=cache_dir, nodesplitter=my_split_by_node)\
        .shuffle(
            int(os.getenv("WDS_SHUFFLE", "128")),
            initial=int(os.getenv("WDS_SHUFFLE_INIT", "128")),
            rng=random.Random(42)
        )\
        .slice(num_samples)\
        .decode("torch")\
        .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
        .to_tuple(*to_tuple)\
        .batched(batch_size, partial=True)\
        .with_epoch(num_worker_batches)

    train_dl_kwargs = dict(batch_size=None, shuffle=False, num_workers=num_workers, pin_memory=False)
    if num_workers > 0:
        train_dl_kwargs.update(prefetch_factor=1, persistent_workers=False)
    train_dl = DataLoader(train_data, **train_dl_kwargs)

    # Compute batch information for the validation stage
    num_batches = max(1, math.floor(num_val / global_batch_size))
    num_worker_batches = num_batches if num_workers <= 1 else max(1, math.floor(num_batches / num_workers))

    print("\nnum_val", num_val)
    print("val_num_batches", num_batches)
    print("val_batch_size", val_batch_size)

    # Build the validation WebDataset pipeline
    val_data = wds.WebDataset(val_url, resampled=False, cache_dir=cache_dir, nodesplitter=my_split_by_node)\
        .decode("torch")\
        .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
        .to_tuple(*to_tuple)\
        .batched(val_batch_size, partial=False)

    val_dl_kwargs = dict(batch_size=None, shuffle=False, num_workers=num_workers, pin_memory=False)
    if num_workers > 0:
        val_dl_kwargs.update(prefetch_factor=1, persistent_workers=False)
    val_dl = DataLoader(val_data, **val_dl_kwargs)

    return train_dl, val_dl, num_train, num_val