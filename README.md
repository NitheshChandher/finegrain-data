# RCDM combined pre-training dataset builder

## What it creates

Given `--out ./dataset`, the script creates:

```text
dataset/
  train/<unique-image-name>.jpg
  val/<unique-image-name>.jpg
  test/<unique-image-name>.jpg
  manifest.csv
  summary.json

dataset_DINOv2/
  train/<unique-image-name>.jpg
  val/<unique-image-name>.jpg
  test/<unique-image-name>.jpg
  DINOv2/train/<unique-image-name>.npy
  DINOv2/val/<unique-image-name>.npy
  DINOv2/test/<unique-image-name>.npy
  manifest.csv

dataset_DINOv3/...
dataset_CLIP/...
```

There are no class folders and no class names in the file names. Image names have the form `<domain>_<running-index>_<hash>.jpg` (for example `birds_000001_baa5f5d460.jpg`), which is globally unique, so different source datasets cannot overwrite each other. The mapping back to class, source, source image ID and original path is kept in `manifest.csv`.

Images are **copied** as real files by default (`--link-mode copy`); symlinks are never created. `--link-mode hardlink` is available if you want to save disk space (the output then shares data with the source files; if the output and source are on different filesystems, it falls back to copying). Note that each `dataset_<REP>` directory contains its own copy of all images, so disk usage is roughly (1 + number of representations) x the size of the image set.

## 1. Obtain the source datasets

### Bird-11K[-CUB]

Use the authors' Bird-11K reconstruction instructions and the `Bird-11K[-CUB]` configuration. The paper describes Bird-11K as reconstructed from existing bird datasets plus Macaulay Library images; it does not simply provide a redistributable archive. Point `--bird-root` at the extracted CUB-excluded image split arranged as:

```text
bird11k_cub_excluded/
  species_one/
    image001.jpg
  species_two/
    image002.jpg
```

If your reconstructed output has a `train/` parent, point `--bird-root` at that `train/` directory, not its parent.

Paper: https://openreview.net/pdf?id=DxEETtLfY3

### CompCars

Download the **web-nature full-car images** from the official page (not the surveillance subset and not the car-parts subset). The official dataset is restricted to non-commercial research use and must not be redistributed.

Official page: https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/

Point `--compcars-root` at the `image/` directory containing:

```text
image/
  <make>/
    <model>/
      <year>/
        image.jpg
```

The default `--compcars-class-depth 2` uses `<make>_<model>` as the class label, ignoring year.

### Military Aircraft

The source repository lists the 6,178-image Military Aircraft Detection Dataset and notes that pre-cropped aircraft images are organized by aircraft type.

Dataset page: https://github.com/matthewgiarra/aircraft-datasets
Kaggle source: https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset

You can download with the Kaggle CLI after configuring your Kaggle credentials:

```bash
pip install kaggle
kaggle datasets download \
  -d a2015003713/militaryaircraftdetectiondataset \
  -p ./raw/military_aircraft --unzip
```

Point `--military-root` at the extracted **pre-cropped class-folder directory**. Expected structure:

```text
military_crops/
  F16/
    image001.jpg
  F35/
    image002.jpg
```

If the Kaggle archive contains full images and Pascal-VOC XML annotations rather than the pre-cropped folders, crop the aircraft using the XML boxes first; this script intentionally does not guess which XML file belongs to which image.

### Open Images V7 aircraft

The script can download Open Images V7 training images and crop detections labelled `Aircraft` or `Airplane` using FiftyOne. Install FiftyOne and pass `--download-oid`. Crops are cached outside the final dataset.

Open Images V7 download documentation:
https://storage.googleapis.com/openimages/web/download_v7.html

Open Images images have individual source licences; check the metadata/licence conditions before redistribution or publication.

## 2. Install dependencies

```bash
python -m pip install \
  torch torchvision transformers pillow numpy pandas \
  fiftyone
```

For optional perceptual-hash leakage checks:

```bash
python -m pip install ImageHash
```

DINOv3 also requires your local DINOv3 repository and checkpoint. DINOv2 uses `torch.hub`; CLIP uses Hugging Face Transformers.

## 3. Run the preparation pipeline

Example using all three representation encoders:

```bash
python prepare_rcdm_combined_dataset.py \
  --bird-root ./raw/bird11k_cub_excluded/train \
  --compcars-root ./raw/compcars/image \
  --military-root ./raw/military_crops \
  --download-oid \
  --oid-cache ./cache/open_images_aircraft \
  --oid-max-images 50000 \
  --out ./dataset \
  --exclude-root ./downstream/FGVC-Aircraft \
  --exclude-root ./downstream/CUB_200_2011 \
  --exclude-root ./downstream/Stanford_Cars \
  --birds 100000 \
  --cars 100000 \
  --aircraft 10000 \
  --val-frac 0.05 \
  --test-frac 0.05 \
  --representations DINOv2 DINOv3 CLIP \
  --dinov3-repo /home/nithesh/Project/dinov3 \
  --dinov3-weights /home/nithesh/Project/rep-ldm/checkpoint/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --batch-size 32 \
  --workers 8 \
  --seed 42
```

The default target is 100K birds + 100K cars + 10K aircraft = **210K images**. The aircraft default is intentionally conservative: the identified conventional-view aircraft sources are much smaller than the bird and car sources. Increase `--aircraft` only after checking how many valid aircraft crops Open Images and the military source actually provide. The script stops rather than silently substituting unrelated images when a requested domain quota cannot be met.

To prepare images only, without extracting embeddings, add `--skip-representations`. You can then rerun representation extraction separately after adjusting the script/configuration if needed.

## 4. Leakage checks

Pass all three downstream dataset roots with `--exclude-root`. Each root should contain every available split (train, validation and test). The script:

1. Excludes exact duplicates using SHA-256.
2. Optionally excludes perceptual near-duplicates using `--phash-exclusion`.
3. Removes exact duplicates across the source datasets.
4. Records source paths and IDs in `manifest.csv`.

Perceptual hashing is conservative and may flag visually similar but non-identical photographs. Review such exclusions if you enable it. The script does **not** automatically use DINOv2 similarity; embedding-based near-duplicate review is a separate, more expensive audit.

This is image-level exclusion. It does not prevent semantic class overlap (e.g. a bird species or aircraft model appearing in both pre-training and downstream datasets). If you need class-disjoint evaluation, that requires an additional class-mapping/exclusion policy.

## 5. Notes

- The split is stratified within each class after sampling.
- The default split is 90% train / 5% validation / 5% test.
- `manifest.csv` maps each image and representation to its source and class.
- DINOv2 and DINOv3 features are the model's image-level output (CLS/global feature). CLIP uses the vision-model CLS token and L2-normalizes it, following the supplied code.
- DINOv2/DINOv3 use the supplied 224x224 center-crop and ImageNet normalization. CLIP uses the CLIP processor's own preprocessing.
- The script creates separate `dataset_<REP>` directories. It does not place the representation folder inside the `dataset/` directory.
- Class balance is still enforced internally (balanced sampling and stratified splitting use the class labels), but the labels are only recorded in `manifest.csv`, not in the folder structure or file names.
