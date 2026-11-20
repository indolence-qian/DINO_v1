"""Paired semantic evidence, intra-image references and region comparison."""

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def load_dinotxt(config, pairs, device):
    backbone_path = Path(config["backbone_weights"]).resolve()
    text_path = Path(config["dinotxt_weights"]).resolve()
    for path in (backbone_path, text_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing DINO weights: {path}")
    repo = Path(__file__).resolve().parents[1]
    model, tokenizer = torch.hub.load(
        str(repo / "dinov3"),
        "dinov3_vitl16_dinotxt_tet1280d20h24l",
        source="local",
        backbone_weights=backbone_path.as_uri(),
        dinotxt_weights=text_path.as_uri(),
        bpe_path_or_url=str(repo / "CLIP" / "bpe_simple_vocab_16e6.txt.gz"),
    )
    # The vendored tokenizer reads vocabulary data only, with no CLIP model loaded.
    model.eval().requires_grad_(False)
    device_dtype = torch.bfloat16 if config["precision"] == "bf16" else torch.float32
    model.text_model.to(device=device, dtype=device_dtype)
    phrases = [row[field] for field in ("normal", "defect") for row in pairs]
    if any(len(tokenizer.encode(phrase)) > 75 for phrase in phrases):
        raise ValueError("An attribute phrase exceeds the dino.txt context length.")
    with torch.no_grad():
        text = model.encode_text(tokenizer.tokenize(phrases).to(device)).float()
    if not torch.isfinite(text).all() or (text.norm(dim=-1) == 0).any():
        raise ValueError("dino.txt returned invalid text embeddings. Check checkpoint compatibility.")
    vision = model.visual_model.to(device)
    return vision, text.reshape(2, len(pairs), -1)


@torch.no_grad()
def normal_references(features, semantic, count, fraction):
    """Farthest point selection within the low-semantic-score subset."""
    batch = torch.arange(features.shape[0], device=features.device)
    candidate_count = max(count, int(features.shape[1] * fraction))
    indices = semantic.topk(candidate_count, largest=False).indices
    candidates = features[batch[:, None], indices]
    next_index = torch.zeros_like(batch)
    closest = features.new_full(indices.shape, float("inf"))
    used = torch.zeros_like(indices, dtype=torch.bool)
    chosen = []
    for _ in range(count):
        chosen.append(next_index)
        used.scatter_(1, next_index[:, None], True)
        seed = candidates[batch, next_index]
        distance = 1.0 - (candidates * seed[:, None]).sum(-1)
        closest = torch.minimum(closest, distance)
        next_index = closest.masked_fill(used, -torch.inf).argmax(1)
    return indices.gather(1, torch.stack(chosen, dim=1))


@torch.no_grad()
def propose_regions(scores, height, width, count, radius):
    rows, cols = torch.meshgrid(
        torch.arange(height, device=scores.device),
        torch.arange(width, device=scores.device), indexing="ij",
    )
    coordinates = torch.stack((rows.flatten(), cols.flatten()), dim=-1)
    proposal_scores = F.avg_pool2d(
        scores.reshape(-1, 1, height, width), 2 * radius + 1,
        stride=1, padding=radius, count_include_pad=False,
    ).flatten(1)
    available = torch.ones_like(proposal_scores, dtype=torch.bool)
    selected = torch.zeros_like(available)
    distances = []
    for _ in range(count):
        allowed = torch.where(available.any(1, keepdim=True), available, ~selected)
        centers = proposal_scores.masked_fill(~allowed, -torch.inf).argmax(1)
        selected.scatter_(1, centers[:, None], True)
        distance = (coordinates[None] - coordinates[centers, None]).abs().amax(dim=-1)
        distances.append(distance)
        available &= distance > 2 * radius
    return coordinates, torch.stack(distances, dim=1)


def region_mean(features, mask):
    weights = mask.float()
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1)
    return weights @ features


class EvidenceHead(nn.Module):
    def __init__(self, config, text_dim, vision_dim, region_llm=None):
        super().__init__()
        self.config = config
        self.region_llm = region_llm
        hidden = config["hidden_dim"]
        self.semantic_log_scale = nn.Parameter(torch.tensor(2.3))
        self.semantic_bias = nn.Parameter(torch.tensor(-1.0))
        self.visual_log_scale = nn.Parameter(torch.tensor(2.3))
        self.visual_bias = nn.Parameter(torch.tensor(-2.0))
        self.global_scale = nn.Parameter(torch.tensor(1.0))
        self.fusion = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.input_projection = nn.Linear(vision_dim, hidden)
        self.attribute_projection = nn.Linear(text_dim, hidden)
        # Equal-capacity region readout for the ROI-only and comparison controls.
        self.roi_readout = nn.Sequential(
            nn.Linear(hidden * 5 + 7, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        if region_llm is not None:
            llm_dim = region_llm.config.hidden_size
            self.llm_bridge = nn.Sequential(nn.Linear(hidden, llm_dim), nn.LayerNorm(llm_dim))
            self.role_embeddings = nn.Parameter(torch.randn(5, llm_dim) * 0.02)
            self.llm_readout = nn.Linear(llm_dim, hidden)
        self.region_classifier = nn.Linear(hidden, 1)
        self.region_correction = nn.Linear(hidden, 1)
        nn.init.zeros_(self.region_correction.weight)
        nn.init.zeros_(self.region_correction.bias)

    def train(self, mode=True):
        super().train(mode)
        if self.region_llm is not None:
            self.region_llm.eval()
        return self

    def forward(self, global_features, patches, raw_patches, text, grid):
        cfg = self.config
        height, width = grid
        batch = torch.arange(patches.shape[0], device=patches.device)
        text = text.float()
        # DINOTxt global representation concatenates CLS and mean-pooled patches.
        patch_text = F.normalize(text[..., -patches.shape[-1]:], dim=-1)
        aligned = F.normalize(patches.float(), dim=-1)
        appearance = F.normalize(raw_patches.float(), dim=-1)
        normal, defect = patch_text
        full_normal, full_defect = F.normalize(text, dim=-1)
        if cfg["pair_mode"] == "shuffled":
            normal = normal.roll(1, dims=0)
            full_normal = full_normal.roll(1, dims=0)
        margins = aligned @ defect.T - aligned @ normal.T
        semantic, _ = margins.max(-1)
        semantic_logits = self.semantic_log_scale.exp().clamp(max=100) * semantic + self.semantic_bias
        fusion_mode = cfg["fusion"]
        visual_only = fusion_mode == "visual" and cfg["region_mode"] == "off"
        reference_ranking = semantic
        if visual_only:
            reference_ranking = 1.0 - F.cosine_similarity(appearance, appearance.mean(1, keepdim=True), dim=-1)
        ref_ids = normal_references(
            appearance, reference_ranking, cfg["prototype_count"], cfg["reference_fraction"]
        )
        refs = appearance[batch[:, None], ref_ids]
        visual = (1.0 - (appearance @ refs.transpose(1, 2)).amax(-1)).clamp(0, 2)
        visual_logits = self.visual_log_scale.exp().clamp(max=100) * visual + self.visual_bias
        disagreement = (semantic_logits.sigmoid() - visual_logits.sigmoid()).abs()
        inputs = torch.stack((semantic_logits, visual_logits, disagreement), dim=-1)
        if fusion_mode == "learned":
            coarse = inputs[..., :2].mean(-1) + self.fusion(inputs).squeeze(-1)
        elif fusion_mode == "fixed":
            coarse = inputs[..., :2].mean(-1)
        else:
            coarse = semantic_logits if fusion_mode == "semantic" else visual_logits

        output = {}
        if not visual_only:
            output["semantic_logits"] = semantic_logits.reshape(-1, 1, height, width)
        region_mode = cfg["region_mode"]
        if region_mode != "off":
            coordinates, distances = propose_regions(
                coarse, height, width, cfg["proposal_count"], cfg["region_radius"]
            )
            region_mask = distances <= cfg["region_radius"]
            context_mask = (distances <= cfg["context_radius"]) & ~region_mask
            roi = region_mean(appearance, region_mask)
            context = region_mean(appearance, context_mask)
            # Prefer a visually similar reference outside the ROI and its surrounding context.
            ref_similarity = F.normalize(roi, dim=-1) @ refs.transpose(1, 2)
            outside = distances.gather(2, ref_ids[:, None].expand(-1, roi.shape[1], -1))
            outside = outside > cfg["context_radius"]
            ref_index = ref_similarity.masked_fill(~outside & outside.any(-1, keepdim=True), -torch.inf)
            ref_index = ref_index.argmax(-1)
            reference = refs[batch[:, None], ref_index]
            reference_ids = ref_ids.gather(1, ref_index)
            attr_margin, attr_index = region_mean(margins, region_mask).max(-1)
            # Roll the raw normal attributes as well, keeping the shuffled control internally consistent.
            normal_attr = text[0].roll(1, dims=0) if cfg["pair_mode"] == "shuffled" else text[0]
            comparison = (
                roi,
                context if region_mode == "compare" else torch.zeros_like(context),
                reference if region_mode == "compare" else torch.zeros_like(reference),
            )
            vision_tokens = self.input_projection(torch.stack(comparison, dim=-2))
            attribute_tokens = self.attribute_projection(torch.stack(
                (normal_attr[attr_index], text[1][attr_index]), dim=-2
            ))
            tokens = torch.cat((vision_tokens, attribute_tokens), dim=-2)
            coord_maps = coordinates.float() / coordinates.new_tensor([height - 1, width - 1])
            centers = region_mean(coord_maps.expand(patches.shape[0], -1, -1), region_mask)
            similarity = F.cosine_similarity(roi, reference, dim=-1)
            if region_mode == "roi":
                # Equal-width input with no direct context or reference feature access.
                similarity = torch.zeros_like(similarity)
            summaries = region_mean(inputs, region_mask)
            region_features = self.roi_readout(torch.cat((
                tokens.flatten(-2), summaries, attr_margin.unsqueeze(-1),
                similarity.unsqueeze(-1), centers,
            ), dim=-1))
            if self.region_llm is not None:
                llm_input = self.llm_bridge(tokens) + self.role_embeddings
                llm_input = llm_input.flatten(0, 1).to(dtype=self.region_llm.dtype)
                # Frozen LLM, but gradients must flow through its input tokens to the bridge.
                llm_output = self.region_llm(inputs_embeds=llm_input, use_cache=False)
                region_features = region_features + self.llm_readout(
                    llm_output.last_hidden_state[:, -1].reshape(*roi.shape[:2], -1)
                    .float()
                    .contiguous()
                )
            region_logits = self.region_classifier(region_features).squeeze(-1)
            correction = self.region_correction(region_features).squeeze(-1).float()
            region_weights = region_mask.float()
            update = (correction[..., None] * region_weights).sum(1)
            coarse = coarse + update / region_weights.sum(1).clamp_min(1)
            output.update(region_logits=region_logits, region_mask=region_mask,
                          region_attributes=attr_index, region_attr_margin=attr_margin,
                          region_references=reference_ids, region_evidence=summaries)

        global_margins = F.normalize(global_features.float(), dim=-1) @ (full_defect - full_normal).T
        local_score = coarse.topk(max(1, (height * width) // 50), dim=-1).values.mean(1)
        image_logits = local_score
        if not visual_only:
            image_logits = image_logits + self.global_scale * global_margins.max(-1).values
        if region_mode != "off":
            image_logits = image_logits + 0.25 * output["region_logits"].max(-1).values
        output.update(pixel_logits=coarse.reshape(-1, 1, height, width), image_logits=image_logits)
        return output


def joint_loss(output, mask, label, weights):
    # Upsampling preserves the gradient from very small defects on the full-size mask.
    shape = mask.shape[-2:]
    logits = F.interpolate(output["pixel_logits"].float(), shape, mode="bilinear", align_corners=False)
    seg_ce = F.binary_cross_entropy_with_logits(logits, mask)
    probability = logits.sigmoid().flatten(1)
    target = mask.flatten(1)
    inter = (probability * target).sum(1)
    overlap = (2 * inter + 1) / (probability.sum(1) + target.sum(1) + 1)
    # BCE already supervises negative images. Dice targets positive images only.
    positive = target.any(-1).float()
    dice = ((1 - overlap) * positive).sum() / positive.sum().clamp_min(1)
    image_ce = F.binary_cross_entropy_with_logits(output["image_logits"].float(), label)
    loss = seg_ce + weights["dice"] * dice + weights["image"] * image_ce
    if "semantic_logits" in output:
        semantic = F.interpolate(output["semantic_logits"].float(), shape, mode="bilinear", align_corners=False)
        loss = loss + weights["semantic"] * F.binary_cross_entropy_with_logits(semantic, mask)
    if "region_logits" in output:
        grid_target = F.adaptive_max_pool2d(mask, output["pixel_logits"].shape[-2:]).flatten(1)
        region_target = (output["region_mask"] & grid_target[:, None].bool()).any(-1).float()
        region_ce = F.binary_cross_entropy_with_logits(output["region_logits"].float(), region_target)
        loss = loss + weights["region"] * region_ce
    return loss
