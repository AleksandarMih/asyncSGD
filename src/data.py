import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

MEAN = [0.4914, 0.4822, 0.4465]
STD  = [0.2023, 0.1994, 0.2010]

_train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def get_loaders(data_root="data", batch_size=128, num_workers=2):
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=_train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=_test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


def visualize_batch(loader, n=32, save_path="outputs/sample_batch.png"):
    images, labels = next(iter(loader))
    images = images[:n]
    labels = labels[:n]

    # Undo normalization for display
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    images = (images * std + mean).clamp(0, 1)

    cols = 8
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 1.6, rows * 1.8))
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.4)

    for i in range(n):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        ax.imshow(images[i].permute(1, 2, 0).numpy())
        ax.set_title(CIFAR10_CLASSES[labels[i]], fontsize=7)
        ax.axis("off")

    fig.suptitle("CIFAR-10 sample batch", fontsize=11, y=1.01)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return save_path
