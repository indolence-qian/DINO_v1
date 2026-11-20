"""Train on labeled source images and evaluate on a separate target dataset."""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .attributes import validate_bank
from .data import InspectionData, source_split
from .metrics import Metrics
from .model import EvidenceHead, joint_loss, load_dinotxt

INDUSTRIAL = ("visa", "mvtec", "btad", "mpdd")
REPO = Path(__file__).resolve().parents[1]


def load_config(path, smoke):
    config = json.loads(path.read_text(encoding="utf-8"))
    for field, override in {
        "source": "SOURCE", "target": "TARGET", "source_root": "SOURCE_ROOT",
        "target_root": "TARGET_ROOT", "backbone_weights": "DINO_WEIGHTS",
        "dinotxt_weights": "DINOTXT_WEIGHTS", "llm_model": "LLM_MODEL",
    }.items():
        if os.environ.get(override):
            config[field] = os.environ[override]
    for field, override in {
        "batch_size": "BATCH_SIZE", "epochs": "EPOCHS", "workers": "WORKERS",
        "accumulation_steps": "ACCUMULATION_STEPS", "image_size": "IMAGE_SIZE",
    }.items():
        if os.environ.get(override):
            config[field] = int(os.environ[override])
    if smoke:
        config.update(epochs=1, batch_size=1, workers=0, accumulation_steps=1)
    return config


def check_config(config):
    source, target = config["source"], config["target"]
    if source not in INDUSTRIAL or target not in INDUSTRIAL or source == target:
        raise ValueError("Choose distinct source and target datasets from visa, mvtec, btad, mpdd.")
    from Datasets import DATASET_REGISTRY

    source_root = Path(config.get("source_root") or DATASET_REGISTRY[source][2])
    target_root = Path(config.get("target_root") or DATASET_REGISTRY[target][2])
    if source_root.resolve() == target_root.resolve():
        raise ValueError("Source and target directories must be different.")
    size = config["image_size"]
    if size < 128 or size % 16 != 0:
        raise ValueError("image_size must be a multiple of 16 and at least 128.")
    tokens = (size // 16) ** 2
    for key in ("proposal_count", "prototype_count"):
        if not 1 <= config[key] <= tokens:
            raise ValueError(f"{key} must be between 1 and the number of patches.")
    if not 0 <= config["region_radius"] < config["context_radius"] < size // 16:
        raise ValueError("Use 0 <= region_radius < context_radius < patch grid side.")
    if not 0 < config["validation_fraction"] < 1 or not 0 < config["reference_fraction"] <= 1:
        raise ValueError("Validation and reference fractions must be positive and at most one.")
    for key, choices in {
        "precision": ("bf16", "fp32"), "fusion": ("learned", "fixed", "semantic", "visual"),
        "region_mode": ("compare", "roi", "off"), "region_engine": ("qwen", "mlp"),
        "pair_mode": ("paired", "shuffled"),
    }.items():
        if config[key] not in choices:
            raise ValueError(f"{key} must be one of {choices}.")
    for key in ("batch_size", "epochs", "accumulation_steps", "hidden_dim"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive.")
    for key in ("backbone_weights", "dinotxt_weights"):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(f"{key} is missing: {config[key]}")


def make_loader(dataset, config, shuffle=False):
    return DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=shuffle,
        num_workers=config["workers"], pin_memory=True,
        persistent_workers=shuffle and config["workers"] > 0,
    )


def build_head(config, text, vision, device):
    region_llm = None
    if config["region_mode"] != "off" and config["region_engine"] == "qwen":
        from transformers import AutoModel

        dtype = torch.bfloat16 if config["precision"] == "bf16" else torch.float32
        region_llm = AutoModel.from_pretrained(
            config["llm_model"], torch_dtype=dtype, use_safetensors=True,
            attn_implementation="sdpa",
        ).requires_grad_(False).eval()
    return EvidenceHead(config, text.shape[-1], vision.backbone.embed_dim, region_llm).to(device)


def predict(vision, head, text, images, use_bf16):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        with torch.no_grad():
            global_features, patches, raw_patches = vision(images)
        grid = (images.shape[-2] // 16, images.shape[-1] // 16)
        if patches.shape[1] != math.prod(grid):
            raise RuntimeError(f"Unexpected number of dino.txt patch tokens: {patches.shape[1]}")
        return head(global_features, patches, raw_patches, text, grid)


def train_epoch(vision, head, text, loader, optimizer, config, device, smoke):
    head.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    steps = min(2, len(loader)) if smoke else len(loader)
    accumulation = config["accumulation_steps"]
    for index, (images, masks, labels, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, dtype=torch.float32, non_blocking=True)
        prediction = predict(vision, head, text, images, config["precision"] == "bf16")
        loss = joint_loss(prediction, masks, labels, config["loss_weights"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at batch {index + 1}.")
        group_size = min(accumulation, steps - (index // accumulation) * accumulation)
        # Weight the last partial batch by its actual sample count.
        group_start = (index // accumulation) * accumulation
        group_samples = min(group_size * loader.batch_size, len(loader.dataset) - group_start * loader.batch_size)
        (loss * (images.shape[0] / group_samples)).backward()
        if (index + 1) % accumulation == 0 or index + 1 == steps:
            torch.nn.utils.clip_grad_norm_(
                (p for p in head.parameters() if p.requires_grad), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total += loss.item()
        if index == 0 or (index + 1) % 50 == 0:
            print(f"  batch {index + 1}/{steps} loss={loss.item():.4f}", flush=True)
        if index + 1 == steps:
            break
    return total / steps


def save_evidence(directory, start, paths, maps, prediction, pairs, threshold, save_maps, stream):
    image_scores = prediction["image_logits"].float().sigmoid().cpu().tolist()
    grid_h, grid_w = prediction["pixel_logits"].shape[-2:]
    regions = prediction.get("region_mask")
    if regions is not None:
        masks = regions.cpu().numpy()
        attrs = prediction["region_attributes"].cpu().numpy()
        references = prediction["region_references"].cpu().numpy()
        margins = prediction["region_attr_margin"].float().cpu().numpy()
        summaries = prediction["region_evidence"].float().cpu().numpy()
        scores = prediction["region_logits"].float().sigmoid().cpu().numpy()
    for offset, (path, anomaly_map, score) in enumerate(zip(paths, maps, image_scores)):
        key = f"{start + offset:05d}"
        record = {"image": path, "image_score": score, "regions": []}
        if save_maps:
            map_name = f"{key}_score.png"
            mask_name = f"{key}_mask.png"
            Image.fromarray(np.rint(anomaly_map * 65535).astype(np.uint16)).save(directory / map_name)
            Image.fromarray(((anomaly_map >= threshold) * 255).astype(np.uint8)).save(directory / mask_name)
            record.update(anomaly_map=map_name, mask=mask_name, pixel_threshold=threshold)
        if regions is not None:
            for index in scores[offset].argsort()[::-1]:
                ys, xs = np.nonzero(masks[offset, index].reshape(grid_h, grid_w))
                y, x = divmod(int(references[offset, index]), grid_w)
                semantic, visual, disagreement = map(float, summaries[offset, index])
                record["regions"].append({
                    "box_xyxy_normalized": [xs.min() / grid_w, ys.min() / grid_h,
                                            (xs.max() + 1) / grid_w, (ys.max() + 1) / grid_h],
                    "normal_reference_xy_normalized": [(x + 0.5) / grid_w, (y + 0.5) / grid_h],
                    "region_score": float(scores[offset, index]),
                    "attribute_pair": pairs[attrs[offset, index]],
                    "attribute_margin": float(margins[offset, index]),
                    "semantic_logit": semantic, "visual_logit": visual, "disagreement": disagreement,
                })
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


@torch.inference_mode()
def evaluate(vision, head, text, loader, config, device, pairs, directory=None, threshold=None,
             save_maps=True):
    head.eval()
    metrics = Metrics(collect_regions=directory is not None)

    def consume(stream=None):
        offset = 0
        for images, masks, labels, paths in loader:
            images = images.to(device, non_blocking=True)
            prediction = predict(vision, head, text, images, config["precision"] == "bf16")
            maps = F.interpolate(
                prediction["pixel_logits"].float(), masks.shape[-2:],
                mode="bilinear", align_corners=False,
            )[:, 0].sigmoid().cpu().numpy()
            metrics.update(maps, masks[:, 0].numpy(), prediction["image_logits"].float().cpu().numpy(),
                           labels.numpy())
            if stream is not None:
                save_evidence(directory, offset, paths, maps, prediction, pairs, threshold, save_maps, stream)
            offset += len(paths)

    if directory is None:
        consume()
    else:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "evidence.jsonl").open("w", encoding="utf-8") as stream:
            consume(stream)
    return metrics.compute()


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "plan_a.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attribute-bank", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="Run two source updates with real weights and data.")
    parser.add_argument("--no-maps", action="store_true", help="Keep metrics and evidence, skip score and mask PNGs.")
    args = parser.parse_args()
    config = load_config(args.config, args.smoke)
    check_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Activate the server's CUDA-enabled PyTorch environment.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support BF16. Set precision to fp32.")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.set_float32_matmul_precision("high")
    bank = json.loads(args.attribute_bank.read_text(encoding="utf-8"))
    pairs = validate_bank(bank)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "config.json", config)
    write_json(args.output / "attribute_bank.json", bank)
    source = InspectionData(config["source"], config["image_size"], config.get("source_root"))
    train_ids, val_ids = source_split(source, config["validation_fraction"], config["seed"])
    write_json(args.output / "source_split.json", {
        "train": [str(source.records[i][2]) for i in train_ids],
        "validation": [str(source.records[i][2]) for i in val_ids],
    })
    print(f"GPU: {torch.cuda.get_device_name(device)}. Source {config['source']}: "
          f"{len(train_ids)} train, {len(val_ids)} validation.", flush=True)
    if args.smoke:
        train_ids = [
            next(i for i in train_ids if source.label(source.records[i][1]) == label)
            for label in (0, 1)
        ]
    train_loader = make_loader(Subset(source, train_ids), config, shuffle=not args.smoke)
    val_loader = make_loader(Subset(source, val_ids), config)
    vision, text = load_dinotxt(config, pairs, device)
    head = build_head(config, text, vision, device)
    trainable = {name: p for name, p in head.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(trainable.values(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    checkpoint = args.output / "best.pth"
    best = float("-inf")
    history = []
    for epoch in range(1, config["epochs"] + 1):
        loss = train_epoch(vision, head, text, train_loader, optimizer, config, device, args.smoke)
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        if args.smoke:
            print(f"Smoke passed: finite loss={loss:.4f}, peak CUDA allocated={peak:.2f} GiB", flush=True)
            return
        metrics = evaluate(vision, head, text, val_loader, config, device, pairs)
        if metrics["I_AUROC"] is None or metrics["P_AUROC"] is None:
            raise ValueError("Source validation requires both image labels and both pixel labels.")
        score = (metrics["I_AUROC"] + metrics["P_AUROC"]) / 2
        history.append(dict(epoch=epoch, loss=loss, peak_cuda_gib=peak, **metrics))
        write_json(args.output / "history.json", history)
        print(f"Epoch {epoch} loss={loss:.4f} source_validation={metrics} "
              f"peak CUDA allocated={peak:.2f} GiB", flush=True)
        if score > best:
            best = score
            temporary = checkpoint.with_suffix(".tmp")
            torch.save({
                "head": {name: parameter.detach().cpu() for name, parameter in trainable.items()},
                "epoch": epoch,
                "threshold": metrics["pixel_threshold"],
            }, temporary)
            temporary.replace(checkpoint)

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["head"].keys() != trainable.keys():
        raise RuntimeError("Checkpoint head does not match the configured model.")
    head.load_state_dict(saved["head"], strict=False)
    from Datasets import DATASET_CLASSES

    target = config["target"]
    if config["pair_mode"] == "shuffled":
        pairs = [dict(row, normal=pairs[i - 1]["normal"]) for i, row in enumerate(pairs)]
    by_class = {}
    for category in sorted(DATASET_CLASSES[target]):
        dataset = InspectionData(target, config["image_size"], config.get("target_root"), category)
        by_class[category] = evaluate(
            vision, head, text, make_loader(dataset, config), config, device, pairs,
            args.output / "target" / category, saved["threshold"], not args.no_maps,
        )
        by_class[category].pop("pixel_threshold")
        print(f"{target}/{category}: {by_class[category]}", flush=True)
    macro = {}
    for name in ("I_AUROC", "P_AUROC", "F1_max", "AUPRO_30"):
        values = [entry[name] for entry in by_class.values() if entry[name] is not None]
        macro[name] = float(np.mean(values)) if values else None
    write_json(args.output / "metrics.json", {
        "source": config["source"], "target": target, "checkpoint_epoch": saved["epoch"],
        "source_pixel_threshold": saved["threshold"], "macro": macro, "by_class": by_class,
    })
    print(f"Final target scores: {macro}. Results: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
