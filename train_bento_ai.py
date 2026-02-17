import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

def main():
    # ===== 設定 =====
    data_dir = Path("dataset")  # dataset/train/ok, dataset/train/ng, dataset/val/ok, dataset/val/ng
    model_name = "efficientnet_b0"
    img_size = 224
    batch_size = 16
    epochs = 10
    lr = 3e-4
    out_path = Path("bento_ai.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # ===== 前処理 =====
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = train_ds.classes
    num_classes = len(class_names)
    print("classes:", class_names)

    # ===== モデル =====
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        train_loss = loss_sum / total
        train_acc = correct / total

        # ---- val ----
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += x.size(0)
        val_acc = correct / total if total else 0.0

        print(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_name": model_name,
                "img_size": img_size,
                "class_names": class_names,
                "state_dict": model.state_dict(),
            }, out_path)
            print("saved:", out_path)

    print("best_val_acc:", best_val_acc)

if __name__ == "__main__":
    main()
