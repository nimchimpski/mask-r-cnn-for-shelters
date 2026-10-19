"""Score saved Mask R-CNN checkpoints using COCO mask AP.

Builds an in-memory COCO-format ground truth from dataset_camps/index.geojson (default val split),
runs each checkpoint over the tiles, and reports the standard COCO mask AP metrics
(AP, AP50, AP75, AP small/medium/large) so runs are comparable to published results.
"""
import argparse
import contextlib
import io
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

sys.path.insert(0, "scripts")
from train_maskrcnn import DATA_DIR, DEVICE, TILE_PX, anchor_gen
from geo_mod import TileDataset

MAX_DETS = 300  # 0.3/0.5 m tiles will exceed torchvision's default 100
OBJ_SCORE_THRESH = 0.6  # operating point for object metrics only; picked from SWEEP_THRESHOLDS (best F1, ~0 count err)
SWEEP_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
CONTACT_SHEET_TILES = 16  # worst tiles by |count error|, at OBJ_SCORE_THRESH


def build_coco_gt(names_list):
    """Wrap TileDataset targets into a COCO-format ground truth object."""
    images, annotations = [], []
    ann_id = 1
    for image_id, name in enumerate(names_list):
        images.append({"id": image_id, "file_name": name, "height": TILE_PX, "width": TILE_PX})
        _, target = TileDataset(names_list).__getitem__(image_id)
        for box, mask in zip(target["boxes"].numpy(), target["masks"].numpy()):
            x0, y0, x1, y1 = box
            rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            annotations.append({
                "id": ann_id, "image_id": image_id, "category_id": 1,
                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                "area": float(mask.sum()), "iscrowd": 0, "segmentation": rle,
            })
            ann_id += 1

    coco_dict = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "structure"}]}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = coco_dict
        coco_gt.createIndex()
    return coco_gt


def load_model(weights_path):
    model = maskrcnn_resnet50_fpn_v2(min_size=TILE_PX, max_size=TILE_PX, box_detections_per_img=MAX_DETS)
    # the v2 builder hardcodes its own anchor generator and ignores it as a kwarg; must set after construction
    model.rpn.anchor_generator = anchor_gen
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_channels_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels_mask, 256, 2)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def run_inference(model, names_list):
    """Returns COCO-format detections (list of dicts) for all input tiles."""
    dataset = TileDataset(names_list)
    detections = []
    with torch.no_grad():
        for image_id, name in enumerate(names_list):
            image, _ = dataset[image_id]
            output = model([image.to(DEVICE)])[0]
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            masks = output["masks"].cpu().numpy()  # (N, 1, H, W) soft masks
            for box, score, mask in zip(boxes, scores, masks):
                binary_mask = (mask[0] >= 0.5).astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(binary_mask))
                rle["counts"] = rle["counts"].decode("ascii")
                x0, y0, x1, y1 = box
                detections.append({
                    "image_id": image_id, "category_id": 1,
                    "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                    "score": float(score), "segmentation": rle,
                })
    return detections


def per_tile_counts(coco_eval, n_images, thresh):
    """Per-image (tp, pred, gt) at a given score threshold, reusing COCO's greedy one-to-one
    matching (thresholding after matching == matching only the kept detections)."""
    tp_arr, pred_arr, gt_arr = [], [], []
    for eval_img in coco_eval.evalImgs[:n_images]:  # first block = areaRng "all"
        if eval_img is None:
            tp_arr.append(0), pred_arr.append(0), gt_arr.append(0)
            continue
        kept_arr = np.array(eval_img["dtScores"]) >= thresh
        matched_arr = eval_img["dtMatches"][0] > 0  # iouThrs[0] = 0.5
        tp_arr.append(int((kept_arr & matched_arr).sum()))
        pred_arr.append(int(kept_arr.sum()))
        gt_arr.append(len(eval_img["gtIds"]))
    return np.array(tp_arr), np.array(pred_arr), np.array(gt_arr)


def print_object_metrics(tp_arr, pred_arr, gt_arr, prefix=""):
    """Object metrics (Gella Eqs. 7-8) from per-tile TP/pred/GT counts."""
    tp, gt = int(tp_arr.sum()), int(gt_arr.sum())
    fp = int((pred_arr - tp_arr).sum())
    fn = gt - tp
    correctness = tp / (tp + fp) if (tp + fp) else float("nan")
    completeness = tp / gt if gt else float("nan")
    quality = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    f1 = 2 * correctness * completeness / (correctness + completeness) if (correctness + completeness) else float("nan")
    count_err_pct = 100 * (tp + fp - gt) / gt if gt else float("nan")
    mae = np.abs(pred_arr - gt_arr).mean()
    print(f"{prefix}TP {tp} FP {fp} FN {fn} (ref {gt})  correctness {correctness:.3f}  "
          f"completeness {completeness:.3f}  quality {quality:.3f}  F1 {f1:.3f}  "
          f"count err {count_err_pct:+.1f}%  tile MAE {mae:.1f}")
    return correctness, completeness, quality, f1, count_err_pct


def save_contact_sheet(names_list, detections, pred_arr, gt_arr, thresh, out_path):
    """Visual QA: red = GT footprints, cyan = kept predictions, worst tiles by |count error|."""
    worst_idx_arr = np.argsort(-np.abs(pred_arr - gt_arr))[:CONTACT_SHEET_TILES]
    dets_by_image = {}
    for det in detections:
        if det["score"] >= thresh:
            dets_by_image.setdefault(det["image_id"], []).append(det)

    n_cols = 4
    n_rows = -(-len(worst_idx_arr) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    for ax, image_id in zip(axes.ravel(), worst_idx_arr):
        name = names_list[image_id]
        with rasterio.open(f"{DATA_DIR}/images/{name}") as src:
            ax.imshow(src.read().transpose(1, 2, 0))
        with rasterio.open(f"{DATA_DIR}/masks/{name}") as src:
            ax.contour(src.read(1) > 0, levels=[0.5], colors="red", linewidths=0.6)
        pred_mask_arr = np.zeros((TILE_PX, TILE_PX), dtype=bool)
        for det in dets_by_image.get(image_id, []):
            pred_mask_arr |= mask_utils.decode(det["segmentation"]).astype(bool)
        ax.contour(pred_mask_arr, levels=[0.5], colors="cyan", linewidths=0.6)
        ax.set_title(f"{name}  pred {pred_arr[image_id]} / ref {gt_arr[image_id]}", fontsize=8)
        ax.axis("off")
    for ax in axes.ravel()[len(worst_idx_arr):]:
        ax.axis("off")
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  contact sheet ({len(worst_idx_arr)} worst tiles by count error) -> {out_path}")


def score_model(label, weights_path, coco_gt, names_list, block_of, split):
    print(f"\n=== {label} ({weights_path}) ===")
    model = load_model(weights_path)
    detections = run_inference(model, names_list)
    if not detections:
        print("  no detections")
        return

    # 1. COCO mask AP over the full score range
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(detections)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
        coco_eval.params.maxDets = [1, 100, MAX_DETS]
        coco_eval.evaluate()
        coco_eval.accumulate()
    coco_eval.summarize()
    # summarize() hardcodes the headline AP row at maxDets=100; recompute at MAX_DETS
    precision_arr = coco_eval.eval["precision"][:, :, 0, 0, -1]  # [IoU, recall, cat, area=all, maxDets=last]
    print(f" AP @[ IoU=0.50:0.95 | area=all | maxDets={MAX_DETS} ] = {precision_arr[precision_arr > -1].mean():.3f}")

    n_images = len(coco_eval.params.imgIds)

    # 2. Score-threshold sweep (object metrics only; COCO AP above already covers the full range)
    print(f"\n  score-threshold sweep @ IoU 0.5:")
    for thresh in SWEEP_THRESHOLDS:
        tp_arr, pred_arr, gt_arr = per_tile_counts(coco_eval, n_images, thresh)
        print_object_metrics(tp_arr, pred_arr, gt_arr, prefix=f"    score>={thresh:.1f}  ")

    # 3. Per-tile / per-block breakdown at the chosen operating point
    tp_arr, pred_arr, gt_arr = per_tile_counts(coco_eval, n_images, OBJ_SCORE_THRESH)
    print(f"\n  object metrics @ IoU 0.5, score >= {OBJ_SCORE_THRESH} (overall, {n_images} tiles):")
    print_object_metrics(tp_arr, pred_arr, gt_arr, prefix="    ")

    blocks_arr = np.array([block_of[name] for name in names_list])
    block_f1_list, block_count_err_list = [], []
    print(f"\n  per-block breakdown ({len(set(blocks_arr))} val blocks):")
    for block in sorted(set(blocks_arr)):
        in_block = blocks_arr == block
        _, _, _, f1, count_err_pct = print_object_metrics(
            tp_arr[in_block], pred_arr[in_block], gt_arr[in_block], prefix=f"    block {block}  ")
        block_f1_list.append(f1)
        block_count_err_list.append(count_err_pct)
    print(f"  block spread: F1 {np.mean(block_f1_list):.3f} +/- {np.std(block_f1_list):.3f} "
          f"(min {min(block_f1_list):.3f}, max {max(block_f1_list):.3f})  "
          f"count err {np.mean(block_count_err_list):+.1f}% +/- {np.std(block_count_err_list):.1f}")

    # 4. Contact sheet of the worst tiles by count error, at the chosen operating point
    save_contact_sheet(names_list, detections, pred_arr, gt_arr, OBJ_SCORE_THRESH,
                        f"tile_comparisons/{label}_{split}_contact_sheet.png")


def main(arg):
    tiles_dict = json.load(open(f"{DATA_DIR}/index.geojson"))["features"]
    tiles_list = [f["properties"]["name"] for f in tiles_dict if f["properties"]["split"] == arg.split]
    block_of = {f["properties"]["name"]: f["properties"]["block"] for f in tiles_dict}
    print(f"scoring on {len(tiles_list)} {arg.split} tiles")
    coco_gt = build_coco_gt(tiles_list)
    score_model(arg.run, arg.ckpt, coco_gt, tiles_list, block_of, arg.split)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])

    main(p.parse_args())
