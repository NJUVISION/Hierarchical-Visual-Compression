DATASET_DICT = {
    "imagenet": {
        "train_label": "labels/imagenet/train.txt", 
        "val_label": "labels/imagenet/val.txt",
        "dataset_path":  "",
        "data_type": "image"
    },
    "openimages": {
        "train_label": "labels/openimages/train.txt", # 9M+ images
        "val_label": "labels/openimages/train.txt",
        "dataset_path":  "",
        "data_type": "image"
    },
    # Processed jsonl + memmap shards (for VAR stage-2 compatible dataset).
    # Path is provided via --proc_dir (preferred) or --data_path[0].
    "proc_memmap": {
        "train_label": "",
        "val_label": "",
        "dataset_path": "",
        "data_type": "proc_memmap"
    }
}