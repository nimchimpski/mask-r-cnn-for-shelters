"""Score saved Mask R-CNN checkpoints on the val split using COCO mask AP.

Builds an in-memory COCO-format ground truth from dataset_camps/index.geojson (val split),
runs each checkpoint over the val tiles, and reports the standard COCO mask AP metrics
(AP, AP50, AP75, AP small/medium/large) so runs are comparable to published results.
"""
import contextlib
import io
import json
import sys

import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

sys.path.insert(0, "scripts")
from train_maskrcnn import DATA_DIR, DEVICE, TILE_PX, TileDataset

MAX_DETS = 300  # 0.3/0.5 m tiles will exceed torchvision's default 100
OBJ_SCORE_THRESH = 0.5  # operating point for object metrics only; COCO AP uses all detections


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
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_channels_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels_mask, 256, 2)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def run_inference(model, names_list):
    """Returns COCO-format detections (list of dicts) for all val tiles."""
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


def score_model(label, weights_path, coco_gt, names_list):
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

    # 2. Object metrics (Gella Eqs. 7-8) at IoU 0.5, reusing COCO's greedy one-to-one matching.
    # Thresholding after greedy-by-score matching equals matching only the kept detections.
    n_images = len(coco_eval.params.imgIds)
    tp_total = fp_total = gt_total = 0
    abs_count_err_list = []
    for eval_img in coco_eval.evalImgs[:n_images]:  # first block = areaRng "all"
        if eval_img is None:
            abs_count_err_list.append(0)
            continue
        kept_arr = np.array(eval_img["dtScores"]) >= OBJ_SCORE_THRESH
        matched_arr = eval_img["dtMatches"][0] > 0  # iouThrs[0] = 0.5
        tp_img = int((kept_arr & matched_arr).sum())
        pred_img = int(kept_arr.sum())
        gt_img = len(eval_img["gtIds"])
        tp_total += tp_img
        fp_total += pred_img - tp_img
        gt_total += gt_img
        abs_count_err_list.append(abs(pred_img - gt_img))

    fn_total = gt_total - tp_total
    correctness = tp_total / (tp_total + fp_total)
    completeness = tp_total / gt_total
    quality = tp_total / (tp_total + fp_total + fn_total)
    f1 = 2 * correctness * completeness / (correctness + completeness)
    count_err_pct = 100 * (tp_total + fp_total - gt_total) / gt_total
    print(f"\n  object metrics @ IoU 0.5, score >= {OBJ_SCORE_THRESH}:")
    print(f"  TP {tp_total}  FP {fp_total}  FN {fn_total}  (ref {gt_total})")
    print(f"  correctness {correctness:.3f}  completeness {completeness:.3f}  "
          f"quality {quality:.3f}  F1 {f1:.3f}")
    print(f"  count error {count_err_pct:+.1f}%  per-tile MAE {np.mean(abs_count_err_list):.1f} "
          f"(mean ref {gt_total / n_images:.1f}/tile)")


def main():
    features_list = json.load(open(f"{DATA_DIR}/index.geojson"))["features"]
    val_names = [f["properties"]["name"] for f in features_list if f["properties"]["split"] == "val"]
    print(f"scoring on {len(val_names)} val tiles")
    coco_gt = build_coco_gt(val_names)

    runs = [
        ("run1_unfrozen_adamw", "maskrcnn_best_run1_unfrozen.pt"),
        ("run2_frozen_adamw", "maskrcnn_best_run2_frozen_adamw.pt"),
        ("run3_frozen_sgd", "maskrcnn_best_run3_sgd.pt"),
    ]
    for label, weights_path in runs:
        score_model(label, weights_path, coco_gt, val_names)


if __name__ == "__main__":
    main()
