import torch
import torchvision.models.segmentation as models

def get_deeplabv3(num_classes=5):
    """Создаёт и настраивает модель DeepLabV3-ResNet50."""
    weights = models.DeepLabV3_ResNet50_Weights.DEFAULT
    model = models.deeplabv3_resnet50(weights=weights)
    # Заменяем последний слой классификации
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    return model

if __name__ == "__main__":
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_deeplabv3().to(device)
    print(model)
