import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import albumentations as A
from pycocotools.coco import COCO

class COCOSegmentationDataset(Dataset):
    """
    Датасет для семантической сегментации COCO.
    Фон = 0, категории = 1..N.
    """
    def __init__(self, images_dir, annotations_file, transform=None):
        self.images_dir = images_dir
        self.coco = COCO(annotations_file)
        self.transform = transform
        self.image_ids = list(self.coco.imgs.keys())
        self.image_info = {img_id: self.coco.imgs[img_id] for img_id in self.image_ids}

    def get_segmentation_mask(self, image_id):
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        image_info = self.coco.imgs[image_id]
        height, width = image_info["height"], image_info["width"]
        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in annotations:
            class_id = ann["category_id"]
            segmentation = ann["segmentation"]
            for seg in segmentation:
                poly = np.array(seg, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(mask, [poly], class_id)
        return mask

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.image_info[img_id]
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        image_pil = Image.open(img_path).convert("RGB")
        image_np = np.array(image_pil, dtype=np.uint8)
        mask_np = self.get_segmentation_mask(img_id)
        if self.transform is not None:
            augmented = self.transform(image=image_np, mask=mask_np)
            image_out = augmented["image"]
            mask_out = augmented["mask"].long()
        else:
            image_out = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.
            mask_out = torch.from_numpy(mask_np).long()
        if image_out.shape[0] != 3:
            raise ValueError(f"Expected 3 channels for image, got {image_out.shape}")
        if mask_out.dim() != 2:
            raise ValueError(f"Expected 2D mask, got {mask_out.shape}")
        if mask_out.dtype != torch.long:
            raise ValueError(f"Mask must be torch.long, got {mask_out.dtype}")
        return image_out, mask_out

    def __len__(self):
        return len(self.image_ids)
