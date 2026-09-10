"""Train-only synchronized Albumentations D4 augmentation."""
import numpy as np


class SynchronizedD4:
    """One D4 symmetry per call, shared across image, label and masks."""

    def __init__(self, seed):
        try:
            import albumentations as A
        except ImportError as exc:
            raise ImportError("albumentations is required for frozen D4 augmentation") from exc
        self.A = A
        self.transform = A.Compose([A.D4(p=1.0)])
        self.set_seed(seed)

    def set_seed(self, seed):
        seed = int(seed)
        if not hasattr(self.transform, "set_random_seed"):
            raise RuntimeError("Albumentations Compose lacks set_random_seed; incompatible version")
        self.transform.set_random_seed(seed)

    def __call__(self, image_chw, label_hw, valid_hw, extra_masks=None):
        image = np.asarray(image_chw, dtype=np.float32)
        label = np.asarray(label_hw)
        valid = np.asarray(valid_hw, dtype=np.uint8)
        extras = [] if extra_masks is None else [np.asarray(x, dtype=np.uint8) for x in extra_masks]
        if image.ndim != 3 or label.ndim != 2 or valid.ndim != 2:
            raise ValueError("Expected CHW image plus HW label/valid")
        if image.shape[1:] != label.shape or label.shape != valid.shape:
            raise ValueError("Spatial shape mismatch for synchronized D4")
        for m in extras:
            if m.ndim != 2 or m.shape != label.shape:
                raise ValueError("Spatial shape mismatch for extra synchronized D4 mask")
        out = self.transform(
            image=np.moveaxis(image, 0, -1),
            masks=[label, valid] + extras,
        )
        image_out = np.moveaxis(np.asarray(out["image"], dtype=np.float32), -1, 0)
        label_out = np.asarray(out["masks"][0])
        valid_out = np.asarray(out["masks"][1], dtype=np.uint8).astype(bool)
        extra_out = [np.asarray(x, dtype=np.uint8).astype(bool) for x in out["masks"][2:]]
        if extra_masks is None:
            return image_out, label_out, valid_out
        return image_out, label_out, valid_out, extra_out
