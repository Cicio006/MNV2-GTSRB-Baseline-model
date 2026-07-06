from torchvision.datasets import GTSRB
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataset = GTSRB(
    root="./data",
    split="train",
    download=True,
    transform=transform
)

test_dataset = GTSRB(
    root="./data",
    split="test",
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)