import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import load_data
from model import get_model
from tqdm import tqdm

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        total += labels.size(0)
        
        pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / total
    epoch_acc = correct.double() / total
    return epoch_loss, epoch_acc.item()

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct.double() / total
    return val_loss, val_acc.item()

def main():
    BATCH_SIZE = 32
    TRAIN_EPOCHS = 5
    FINE_TUNE_EPOCHS = 3
    DATA_DIR = "./data"
    WEIGHTS_DIR = "./model/weights"

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    device = torch.device("cpu")

    train_loader, test_loader, class_names = load_data(data_dir=DATA_DIR, batch_size=BATCH_SIZE)

    model = get_model(classes=len(class_names), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    save_path = os.path.join(WEIGHTS_DIR, "model.pth")

    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    for epoch in range(TRAIN_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch ({epoch + 1}/{TRAIN_EPOCHS}) | TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(WEIGHTS_DIR, "model.pth")
            torch.save({"model_state_dict" : model.state_dict(), 
                        "class_names" : class_names}, save_path)

    for param in model.parameters():
        param.requires_grad = True

    # fine-tune
    # You, now I see, keeping everything inside
    # With youuuuuuuuuuuu
    # You, now I see, even when I close my eyes
    # WITH YOUUUUUUUUUUUUUUUUUUUU
    # ladies and gentlemen, Mr. Hahn!

    optimizer_ft = optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(3):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer_ft, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch ({epoch + 1}/{FINE_TUNE_EPOCHS}) | TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state_dict" : model.state_dict(), 
                        "class_names" : class_names}, save_path)

if __name__ == "__main__":
    main()
