import utils


def build_dataloaders(args, accelerator, num_devices, num_workers):
    """
    Build training and validation dataloaders for multiple subjects,
    and return the corresponding sample count information.
    """
    train_dls = {}
    val_dls = {}
    num_train_samples = {}
    num_val_samples = {}

    # Fixed numbers of training and validation samples for each subject
    FIXED_NUM_TRAIN = 8859
    FIXED_NUM_VAL = 982

    accelerator.print('\n--- Preparing NSD WebDataset Data for Multi-Subject Training ---')

    for subj in args.subjects:
        # Load data for each subject one by one
        accelerator.print(f"Loading data for subject {subj}...")

        # Construct the training set path:
        # the training set is composed of both train shards and the val shard,
        # using a brace pattern for WebDataset expansion
        train_url = "{" + f"{args.data_path}/webdataset_avg_split/train/train_subj0{subj}_" + "{0..17}.tar," + \
                    f"{args.data_path}/webdataset_avg_split/val/val_subj0{subj}_0.tar" + "}"

        # Construct the validation set path: this actually uses the test shards
        val_url = f"{args.data_path}/webdataset_avg_split/test/test_subj0{subj}_" + "{0..1}.tar"

        # Construct the metadata file path for the current subject
        meta_url = f"{args.data_path}/webdataset_avg_split/metadata_subj0{subj}.json"

        # Call the general-purpose dataloader builder in utils
        # to create the training and validation dataloaders for the current subject
        train_dl, val_dl, n_train, n_val = utils.get_dataloaders(
            args.batch_size, 'images',
            num_devices=num_devices,
            num_workers=num_workers,
            train_url=train_url,
            val_url=val_url,
            meta_url=meta_url,
            num_train=FIXED_NUM_TRAIN,
            num_val=FIXED_NUM_VAL,
            val_batch_size=32,
            cache_dir=args.data_path,
            voxels_key='nsdgeneral.npy',
            to_tuple=["voxels", "images"],
            subj=subj,
        )

        # Store the dataloaders and sample counts by subject ID
        train_dls[subj] = train_dl
        val_dls[subj] = val_dl
        num_train_samples[subj] = n_train
        num_val_samples[subj] = n_val

    return train_dls, val_dls, num_train_samples, num_val_samples