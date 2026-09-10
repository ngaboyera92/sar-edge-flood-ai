"""Train-only synchronized Albumentations D4 augmentation."""
import numpy as np


class SynchronizedD4:
    """One D4 symmetry per call, shared across image, label and valid mask."""

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

    def __call__(self, image_chw, label_hw, valid_hw):
        image = np.asarray(image_chw, dtype=np.float32)
        label = np.asarray(label_hw)
        valid = np.asarray(valid_hw, dtype=np.uint8)
        if image.ndim != 3 or label.ndim != 2 or valid.ndim != 2:
            raise ValueError("Expected CHW image plus HW label/valid")
        if image.shape[1:] != label.shape or label.shape != valid.shape:
            raise ValueError("Spatial shape mismatch for synchronized D4")
        out = self.transform(
            image=np.moveaxis(image, 0, -1),
            masks=[label, valid],
        )
        image_out = np.moveaxis(np.asarray(out["image"], dtype=np.float32), -1, 0)
        label_out = np.asarray(out["masks"][0])
        valid_out = np.asarray(out["masks"][1], dtype=np.uint8).astype(bool)
        return image_out, label_out, valid_out
