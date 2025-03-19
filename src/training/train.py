import os
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Импортируем наш датасет, модель, логгер и конфигурацию
from src.datasets.custom_dataset import COCOSegmentationDataset
from models.deeplabv3 import get_deeplabv3
from src.utils.logger import Logger
from src.config import IMAGES_DIR, ANNOTATIONS_FILE, CHECKPOINT_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE, DEVICE, NUM_CLASSES

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LOG_FILE = os.path.join('logs', 'training_logs.txt')
logger = Logger(LOG_FILE)

def main():
    try:
        logger.log(f"Using device: {DEVICE}")

        # Определяем аугментации для обучения
        train_transform = A.Compose([
            A.Resize(1080, 1920),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={"mask": "mask"})

        # Загружаем датасет
        dataset = COCOSegmentationDataset(
            images_dir=IMAGES_DIR,
            annotations_file=ANNOTATIONS_FILE,
            transform=train_transform
        )

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )

        model = get_deeplabv3(num_classes=NUM_CLASSES).to(DEVICE)
        logger.log("Model loaded and moved to device")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader)*EPOCHS)

        scaler = GradScaler()
        writer = SummaryWriter()
        best_loss = float("inf")

        for epoch in range(EPOCHS):
            try:
                model.train()
                epoch_loss = 0.0
                logger.log(f"\nEpoch {epoch+1}/{EPOCHS}")

                last_outputs = None
                last_images = None

                for batch_idx, (images, masks) in enumerate(dataloader):
                    try:
                        if batch_idx == 0:
                            logger.log(f"  [DEBUG] images.shape={images.shape}, dtype={images.dtype}")
                            logger.log(f"  [DEBUG] masks.shape={masks.shape}, dtype={masks.dtype}")

                        images = images.to(DEVICE, non_blocking=True)
                        masks = masks.to(DEVICE, non_blocking=True)

                        optimizer.zero_grad()

                        with autocast(device_type="cuda"):
                            outs = model(images)["out"]
                            loss = criterion(outs, masks)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()

                        epoch_loss += loss.item()
                        logger.log(f"  batch {batch_idx+1}/{len(dataloader)}: loss={loss.item():.4f}")

                        last_outputs = outs
                        last_images = images

                    except Exception as batch_err:
                        logger.log(f"Error in batch {batch_idx+1}:\n{traceback.format_exc()}")

                avg_loss = epoch_loss / len(dataloader)
                logger.log(f"Epoch {epoch+1} done. Avg Loss={avg_loss:.4f}")
                writer.add_scalar("Loss/train", avg_loss, epoch)

                if epoch % 5 == 0 and last_outputs is not None:
                    pred_mask = torch.argmax(last_outputs, dim=1, keepdim=True)
                    writer.add_images("Input Images", last_images, epoch)
                    writer.add_images("Predicted Masks", pred_mask, epoch)

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
                    logger.log(f"Saved best model. best_loss={best_loss:.4f}")

                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"deeplabv3_epoch{epoch+1}.pth"))

            except Exception as epoch_err:
                logger.log(f"Error in epoch {epoch+1}:\n{traceback.format_exc()}")

        writer.close()
        logger.log("🎉 Training completed!")

    except Exception as main_err:
        logger.log(f"Critical error:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
