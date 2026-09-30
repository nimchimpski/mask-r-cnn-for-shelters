import json
import os
import time

import argparse
import csv
import numpy as np
import rasterio
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from scripts.geo_mod import TileDataset, DATA_DIR



# DATA_DIR = "dataset_camps"
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
# TRAINABLE_BACKBONE_LAYERS = 0  # 0 = freeze whole ResNet; torchvision default is 3 (only stem + layer1 frozen)
SEED = 0
# CHECKPOINT = "checkpoints/maskrcnn_checkpoint.pt"  # {epoch, model, optimizer} - lets an interrupted run resume

anchor_gen = AnchorGenerator(sizes=((16,), (32,), (64,), (128,), (256,)),
                             aspect_ratios=((0.5, 1.0, 2.0),) * 5)



# class TileDataset(Dataset):
#     """torch's training loop needs a Dataset subclass. Masks already carry per-structure
#     ids from the source shapefile (see build_dataset.py), so instances don't need
#     reconstructing from a merged blob — touching structures stay separate."""

#     # def __init__(self, names_list, augment=False):
#     #     self.names_list = names_list
#     #     self.augment = augment

#     def __init__(self, names_list, augment=False, aug="flip", min_area=4):
#         self.names_list, self.augment, self.aug, self.min_area = names_list, augment, aug, min_area


#     def __len__(self):
#         return len(self.names_list)

#     def __getitem__(self, i):
#         name = self.names_list[i]
#         with rasterio.open(f"{DATA_DIR}/images/{name}") as s:
#             img_arr = s.read()
#         with rasterio.open(f"{DATA_DIR}/masks/{name}") as s:
#             instance_arr = s.read(1)
#         image = torch.from_numpy(img_arr).float() / 255.0

#         if self.augment:
#             if np.random.random() < 0.5:
#                 image = image.flip(-1); instance_arr = instance_arr[:, ::-1]
#             if np.random.random() < 0.5:
#                 image = image.flip(-2); instance_arr = instance_arr[::-1, :]
#             if self.aug == "full":
#                 k = np.random.randint(4)
#                 image = torch.rot90(image, k, dims=(-2, -1))
#                 instance_arr = np.rot90(instance_arr, k)
#                 c = np.random.uniform(0.85, 1.15)   # contrast
#                 b = np.random.uniform(-0.08, 0.08)  # brightness, same shift on all bands
#                 m = image.mean()
#                 image = ((image - m) * c + m + b).clamp(0, 1)
#             instance_arr = np.ascontiguousarray(instance_arr)

#         boxes, masks = [], []
#         for instance_id in np.unique(instance_arr):
#             if instance_id == 0:
#                 continue
#             mask_k = instance_arr == instance_id
#             ys, xs = np.where(mask_k)
#             if len(xs) < self.min_area:
#                 continue
#             boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
#             masks.append(mask_k)

#         masks_arr = np.stack(masks) if masks else np.zeros((0, *instance_arr.shape), dtype=bool)
#         boxes_arr = np.array(boxes, dtype=np.float32).reshape(-1, 4)
#         target = {
#             "boxes": torch.from_numpy(boxes_arr),
#             "labels": torch.ones(len(boxes), dtype=torch.int64),
#             "masks": torch.from_numpy(masks_arr.astype(np.uint8)),
#             "image_id": torch.tensor([i]),
#         }
#         return image, target


def main(args):

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    run = args.run
    ckpt_path = f"checkpoints/{run}_checkpoint.pt"
    os.makedirs("checkpoints", exist_ok=True); os.makedirs("logs", exist_ok=True)

    # 1. splits from the dataset index (plain json: geopandas isn't installed in this env)
    features_list = json.load(open(f"{DATA_DIR}/index.geojson"))["features"]
    names_by_split = {"train": [], "val": [], "test": []}
    for feat in features_list:
        names_by_split[feat["properties"]["split"]].append(feat["properties"]["name"])
    if args.train_frac < 1.0:
        # fixed RNG independent of SEED so 25%/50% subsets are nested prefixes of the full train set
        order = np.random.RandomState(0).permutation(len(names_by_split["train"]))
        keep_n = round(len(order) * args.train_frac)
        names_by_split["train"] = [names_by_split["train"][i] for i in order[:keep_n]]
    print({k: len(v) for k, v in names_by_split.items()})

    collate_fn = lambda batch: tuple(zip(*batch))
    train_loader = DataLoader(TileDataset(names_by_split["train"], augment=True, aug=args.aug, min_area=args.min_area),
                               batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(TileDataset(names_by_split["val"], min_area=args.min_area), batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_fn)

    # 2. Mask R-CNN, pretrained backbone, 2 classes: background + structure
    # default transform resizes to 800px; TRAIN_SCALES keeps chips at (jittered) native size
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT, 
                                     min_size=TRAIN_SCALES, 
                                     max_size=TILE_PX, 
                                     trainable_backbone_layers=args.backbone_layers, 
                                     box_detections_per_img=200)
    # the v2 builder hardcodes its own anchor generator, so it can't be passed as a kwarg
    model.rpn.anchor_generator = anchor_gen
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
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        print(f"resumed from {ckpt_path}: starting at epoch {start_epoch}", flush=True)

    warmup = None
    # LinearLR scales the LR on construction, so it must be created once, not per epoch
    if start_epoch == 0:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, total_iters=min(1000, len(train_loader)))
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
            if epoch == 0 and warmup is not None:
                warmup.step()
            epoch_loss += loss.item()
            if step % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (step + 1) * (len(train_loader) - step - 1)
                print(f"  epoch {epoch + 1}/{EPOCHS} step {step}/{len(train_loader)}: loss {loss.item():.3f} "
                      f"| lr {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.0f}s elapsed, ~{eta:.0f}s left in epoch", flush=True)
        scheduler.step()

        # model.train() mode still returns losses (not predictions); no_grad just skips backprop
        print(f"  validating on {len(val_loader)} batches...", flush=True)
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
            torch.save(model.state_dict(), f"checkpoints/{run}_bestloss.pt")
            print(f"  new best val loss -> saved {run}_bestloss.pt", flush=True)
        ckpt_state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "best_val": best_val}
        torch.save(ckpt_state, ckpt_path)  # for resuming interrupted training

    torch.save(model.state_dict(), f"checkpoints/{run}_final.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--backbone-layers", type=int, default=0)
    p.add_argument("--aug", choices=["flip", "full"], default="flip")
    p.add_argument("--min-area", type=int, default=4)
    p.add_argument("--train-frac", type=float, default=1.0)
    main(p.parse_args())
