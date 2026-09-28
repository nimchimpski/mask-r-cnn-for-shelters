import json
import os
import time

import numpy as np
import rasterio
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor




DATA_DIR = "dataset_camps"
# torch 2.7/torchvision 0.22 hung on MPS in Mask R-CNN's RoIAlign backward;
# fixed in torch 2.14/torchvision 0.29 (env: camp_maskrcnn) - use that env, not floodai_train
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TILE_PX = 512  # must match build_dataset.py
# scale jitter (He et al. 2017 sec 3.1): random train-time scale, native size at eval (last entry)
TRAIN_SCALES = (round(TILE_PX * 0.8), round(TILE_PX * 0.9), TILE_PX)
BATCH_SIZE = 2
EPOCHS = 15  # ~7 min/epoch at 512px, batch 2, on this M2
# He et al.: lr 0.02 at batch 16 (8 GPUs x 2/GPU); linearly scaled to our batch size
LR = 0.02 * BATCH_SIZE / 16
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LR_DECAY_EPOCHS = [int(EPOCHS * 0.75), int(EPOCHS * 0.9)]  # 10x drop at each, as in the paper
TRAINABLE_BACKBONE_LAYERS = 0  # 0 = freeze whole ResNet; torchvision default is 3 (only stem + layer1 frozen)
SEED = 0
CHECKPOINT = "maskrcnn_checkpoint.pt"  # {epoch, model, optimizer} - lets an interrupted run resume


class TileDataset(Dataset):
    """torch's training loop needs a Dataset subclass. Masks already carry per-structure
    ids from the source shapefile (see build_dataset.py), so instances don't need
    reconstructing from a merged blob — touching structures stay separate."""

    def __init__(self, names_list, augment=False):
        self.names_list = names_list
        self.augment = augment

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, i):
        name = self.names_list[i]
        with rasterio.open(f"{DATA_DIR}/images/{name}") as s:
            img_arr = s.read()  # (3, H, W) uint8
        with rasterio.open(f"{DATA_DIR}/masks/{name}") as s:
            instance_arr = s.read(1)  # (H, W) uint16, 0 = background, 1..N = instance id

        boxes, masks = [], []
        for instance_id in np.unique(instance_arr):
            if instance_id == 0:
                continue
            mask_k = instance_arr == instance_id
            ys, xs = np.where(mask_k)
            if len(xs) < 4:  # drop specks too small to be a real structure
                continue
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            masks.append(mask_k)

        image = torch.from_numpy(img_arr).float() / 255.0
        masks_arr = np.stack(masks) if masks else np.zeros((0, *instance_arr.shape), dtype=bool)
        boxes_arr = np.array(boxes, dtype=np.float32).reshape(-1, 4)

        if self.augment:
            # overhead imagery has no canonical orientation, so flips are safe augmentation
            h, w = instance_arr.shape
            if np.random.random() < 0.5:  # horizontal flip
                image = image.flip(-1)
                masks_arr = masks_arr[:, :, ::-1]
                boxes_arr[:, [0, 2]] = w - boxes_arr[:, [2, 0]]
            if np.random.random() < 0.5:  # vertical flip
                image = image.flip(-2)
                masks_arr = masks_arr[:, ::-1, :]
                boxes_arr[:, [1, 3]] = h - boxes_arr[:, [3, 1]]

        target = {
            "boxes": torch.from_numpy(boxes_arr.copy()),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "masks": torch.from_numpy(masks_arr.copy().astype(np.uint8)),
            "image_id": torch.tensor([i]),
        }
        return image, target


def main():

    torch.manual_seed(SEED)

    # 1. splits from the dataset index (plain json: geopandas isn't installed in this env)
    features_list = json.load(open(f"{DATA_DIR}/index.geojson"))["features"]
    names_by_split = {"train": [], "val": [], "test": []}
    for feat in features_list:
        names_by_split[feat["properties"]["split"]].append(feat["properties"]["name"])
    print({k: len(v) for k, v in names_by_split.items()})

    collate_fn = lambda batch: tuple(zip(*batch))
    train_loader = DataLoader(TileDataset(names_by_split["train"], augment=True), batch_size=BATCH_SIZE,
                               shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(TileDataset(names_by_split["val"]), batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_fn)

    # 2. Mask R-CNN, pretrained backbone, 2 classes: background + structure
    # default transform resizes to 800px; TRAIN_SCALES keeps chips at (jittered) native size
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT, min_size=TRAIN_SCALES, max_size=TILE_PX,
                                     trainable_backbone_layers=TRAINABLE_BACKBONE_LAYERS)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_channels_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels_mask, 256, 2)
    model.to(DEVICE)

    # 3. train (resume from a checkpoint if this run was interrupted)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=LR,
                                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_DECAY_EPOCHS, gamma=0.1)
    start_epoch = 0
    best_val = float("inf")
    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        print(f"resumed from {CHECKPOINT}: starting at epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        # v2 backbone uses plain BatchNorm; batch-of-2 stats would overwrite the pretrained running stats
        for mod in model.backbone.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.eval()
        epoch_loss = 0.0
        t0 = time.time()
        for step, (images, targets) in enumerate(train_loader):
            images = [im.to(DEVICE) for im in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            # MPS out-of-memory errors don't raise; they surface as NaN losses that then corrupt the weights
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch} step {step}: {loss_dict}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)}: loss {loss.item():.3f} "
                      f"({time.time() - t0:.0f}s elapsed)", flush=True)
        scheduler.step()

        # model.train() mode still returns losses (not predictions); no_grad just skips backprop
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [im.to(DEVICE) for im in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                val_loss += sum(model(images, targets).values()).item()

        print(f"epoch {epoch}: train loss {epoch_loss / len(train_loader):.3f} | "
              f"val loss {val_loss / len(val_loader):.3f} | lr {scheduler.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s", flush=True)
        if val_loss / len(val_loader) < best_val:
            best_val = val_loss / len(val_loader)
            torch.save(model.state_dict(), "maskrcnn_best.pt")
            print(f"  new best val loss -> saved maskrcnn_best.pt", flush=True)
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "best_val": best_val}, CHECKPOINT)

    torch.save(model.state_dict(), "maskrcnn_structures.pt")


if __name__ == "__main__":
    main()
