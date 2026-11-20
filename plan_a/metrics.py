"""Streaming pixel metrics to keep evaluation memory bounded."""

import numpy as np
from scipy.ndimage import label as connected_components
from sklearn.metrics import roc_auc_score


def area(x, y):
    return float(np.dot(np.diff(x), (y[1:] + y[:-1]) * 0.5))


class Metrics:
    def __init__(self, bins=4096, collect_regions=True):
        self.bins = bins
        self.positive = np.zeros(bins, dtype=np.int64)
        self.negative = np.zeros_like(self.positive)
        self.region_overlap = np.zeros(bins, dtype=np.float64)
        self.collect_regions = collect_regions
        self.region_count = 0
        self.image_scores = []
        self.image_labels = []

    def update(self, scores, masks, image_scores, labels):
        if not np.isfinite(scores).all() or not np.isfinite(image_scores).all():
            raise FloatingPointError("Non-finite anomaly score.")
        self.image_scores.extend(image_scores.tolist())
        self.image_labels.extend(labels.tolist())
        # The same quantization applies to all images and all classes.
        binned = np.floor(scores * (self.bins - 1)).astype(np.int32)
        for prediction, mask in zip(binned, masks):
            self.positive += np.bincount(prediction[mask], minlength=self.bins)
            self.negative += np.bincount(prediction[~mask], minlength=self.bins)
            if self.collect_regions and mask.any():
                regions, count = connected_components(mask, structure=np.ones((3, 3), dtype=bool))
                region_ids = regions[mask]
                sizes = np.bincount(region_ids)
                self.region_count += count
                self.region_overlap += np.bincount(
                    prediction[mask], weights=1.0 / sizes[region_ids], minlength=self.bins
                )

    def compute(self):
        result = dict(I_AUROC=None, P_AUROC=None, F1_max=None, AUPRO_30=None, pixel_threshold=None)
        if set(self.image_labels) == {0, 1}:
            result["I_AUROC"] = float(roc_auc_score(self.image_labels, self.image_scores))
        positives = self.positive.sum()
        negatives = self.negative.sum()
        if positives == 0 or negatives == 0:
            return result
        tp = np.cumsum(self.positive[::-1])
        fp = np.cumsum(self.negative[::-1])
        fpr = np.r_[0, fp / negatives]
        result["P_AUROC"] = area(fpr, np.r_[0, tp / positives])
        f1 = 2 * tp / (positives + tp + fp)
        best = int(f1.argmax())
        result["F1_max"] = float(f1[best])
        result["pixel_threshold"] = (self.bins - 1 - best) / (self.bins - 1)
        if self.region_count:
            pro = np.r_[0, np.cumsum(self.region_overlap[::-1]) / self.region_count]
            cap = 0.30
            end = int(np.searchsorted(fpr, cap, side="right"))
            pro_cap = float(np.interp(cap, fpr[end - 1:end + 1], pro[end - 1:end + 1]))
            result["AUPRO_30"] = area(np.r_[fpr[:end], cap], np.r_[pro[:end], pro_cap]) / cap
        return result
