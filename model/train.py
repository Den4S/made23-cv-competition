from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models
from torchvision import transforms
from sklearn.metrics import f1_score


class ImagesDataset(Dataset):
    def __init__(self, data_dir, data_df, split='train',
                 transform=transforms.Compose([transforms.ToTensor()]), target_transform=None):
        self.split = split
        split_df = None
        if self.split == 'train':
            split_df = data_df[data_df['val'] == 0]
        elif self.split == 'val':
            split_df = data_df[data_df['val'] == 1]
        elif self.split == 'test':
            split_df = data_df

        self.data_dir = data_dir
        self.files = split_df.image_id.to_numpy()

        if self.split in ('train', 'val'):
            self.labels = split_df.num_label.to_numpy()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.data_dir / self.files[idx]
        image = Image.open(file_path).convert('RGB')

        sample = None
        if self.split in ('train', 'val'):
            label = self.labels[idx]
            if self.target_transform:
                label = self.target_transform(label)
            sample = self.transform(image), label
        elif self.split == 'test':
            sample = (self.transform(image),)

        return sample


class ScaleAndPadToSquare:
    def __init__(self, output_size=224):
        self.output_size = output_size

    def __call__(self, img):
        old_size = img.size  # old_size (width, height)
        ratio = float(self.output_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        # resize the input image
        img = img.resize(new_size, Image.ANTIALIAS)
        # create a new image and paste the resized on it
        new_img = Image.new('RGB', (self.output_size, self.output_size))
        new_img.paste(
            img, ((self.output_size - new_size[0]) // 2, (self.output_size - new_size[1]) // 2)
        )
        return new_img


def train(model, loader, criterion, optimizer, device, scheduler=None):
    model.train()
    train_loss = []
    train_f1_score = []
    for batch in tqdm(loader, total=len(loader), desc="training...", position=0, leave=True):
        images = batch[0].to(device)  # B x 3 x SQUARE_SIZE x SQUARE_SIZE
        labels = batch[1].to(device)  # B x NUM_CLASSES
        with autocast():
            pred_logits = model(images)  # B x NUM_CLASSES
            loss = criterion(pred_logits, labels)  # loss(input, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pred_labels = torch.argmax(pred_logits.detach(), dim=1).cpu().numpy()
        f1 = f1_score(y_true=labels.detach().cpu().numpy(), y_pred=pred_labels, average='micro')

        train_loss.append(loss.item())
        train_f1_score.append(f1)
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()

    return np.mean(train_loss), np.mean(train_f1_score)


def validate(model, loader, criterion, device):
    model.eval()
    val_loss = []
    val_f1_score = []
    for batch in tqdm(loader, total=len(loader), desc="validation...", position=0, leave=True):
        images = batch[0].to(device)
        labels = batch[1].to(device)
        with torch.no_grad():
            pred_logits = model(images)
            loss = criterion(pred_logits, labels)  # loss(input, target)
            pred_labels = torch.argmax(pred_logits, dim=1).cpu().numpy()
            f1 = f1_score(y_true=labels.cpu().numpy(), y_pred=pred_labels, average='micro')

        val_loss.append(loss.item())
        val_f1_score.append(f1)
    return np.mean(val_loss), np.mean(val_f1_score)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    DATA_PATH = Path('./vk-made-sports-image-classification')

    train_df = pd.read_csv(DATA_PATH / 'train.csv')
    test_df = pd.read_csv(DATA_PATH / 'test.csv')
    labels = train_df.label.value_counts().index.tolist()
    label2number = {k: v for v, k in enumerate(labels)}
    number2label = {v: k for k, v in label2number.items()}

    print(f"train DF samples: {train_df.shape[0]}")

    NUM_CLASSES = len(label2number)
    SAVE_PATH = Path("./models")
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    SQUARE_SIZE = 320  # 224 320 360 480
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    MODEL_NAME = "efficientnet_v2_s"
    TRAIN_SIZE = 0.85
    NUM_EPOCHS = 20  # 20 30 40
    BATCH_SIZE = 32  # 32 64 128 256
    LEARNING_RATE = 1e-4  # 1e-3 5e-4 3e-4 1e-4
    TRANSFER = True

    train_files = [DATA_PATH / 'train' / id for id in train_df.image_id]
    train_df['num_label'] = train_df.label.map(label2number)
    test_files = [DATA_PATH / 'test' / id for id in test_df.image_id]

    # Train/val split inside train_df
    train_df['val'] = (np.random.rand(len(train_df)) > TRAIN_SIZE) * 1
    print(f"validation samples count: {train_df['val'].sum()}")
    print(train_df.loc[train_df.val == 1, 'label'].value_counts())

    train_transforms = transforms.Compose([
        ScaleAndPadToSquare(SQUARE_SIZE),
        transforms.RandomGrayscale(p=0.3),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.RandomApply(
            transforms=[transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.2, 2))], p=0.3
        ),
        transforms.RandomRotation(degrees=(-5, 5)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    test_transforms = transforms.Compose([
        ScaleAndPadToSquare(SQUARE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    train_dataset = ImagesDataset(
        data_dir=DATA_PATH / 'train',
        data_df=train_df,
        split='train',
        transform=train_transforms
    )

    val_dataset = ImagesDataset(
        data_dir=DATA_PATH / 'train',
        data_df=train_df,
        split='val',
        transform=test_transforms
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        num_workers=4, pin_memory=True,
        shuffle=True, drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        num_workers=4, pin_memory=True,
        shuffle=False, drop_last=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Creating model...")

    model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(model.classifier[1].in_features, NUM_CLASSES, bias=True)
    )

    model.to(device)
    print(f"{count_parameters(model)} trainable params")
    if TRANSFER:
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.features[-2:].parameters():  # (6): Sequential, (7): Conv2dNormActivation
            p.requires_grad = True
        # for p in model.avgpool.parameters():
        #     p.requires_grad = True  # by default
        # for p in model.classifier.parameters():
        #     p.requires_grad = True  # by default
        print(f"Transfer learning: {count_parameters(model)} trainable params")

    scaler = GradScaler()  # to decrease GPU memory usage

    # n_samples / (n_classes * np.bincount(y))
    class_weights = len(train_df) / (NUM_CLASSES * train_df.label.value_counts().to_numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction="mean")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, amsgrad=True, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS,
                                              max_lr=LEARNING_RATE, final_div_factor=1e3)
    best_val_loss = np.inf

    for epoch in range(NUM_EPOCHS):
        train_ce_loss, train_f1 = train(model, train_loader, criterion, optimizer, device, scheduler=scheduler)
        val_ce_loss, val_f1 = validate(model, val_loader, criterion, device=device)
        log_df = pd.DataFrame(
            {'train_ce': [train_ce_loss], 'val_ce': [val_ce_loss], 'train_f1': [train_f1], 'val_f1': [val_f1]}
        )
        if epoch == 0:
            log_df.to_csv(SAVE_PATH / f'{MODEL_NAME}_train_log.csv', mode='w', header=True, index=False)
        else:
            log_df.to_csv(SAVE_PATH / f'{MODEL_NAME}_train_log.csv', mode='a', header=False, index=False)
        print(f"Epoch #{epoch + 1:<{2}}")
        print(f"train CE loss: {train_ce_loss:.3f}  val CE loss: {val_ce_loss:.3f}")
        print(f"train F1: {train_f1:.3f}  val F1: {val_f1:.3f}")
        if val_ce_loss < best_val_loss:
            best_val_loss = val_ce_loss
            with open(SAVE_PATH / f"{MODEL_NAME}_ep{epoch+1}_t{train_ce_loss:.2f}_v{val_ce_loss:.2f}.pth", "wb") as fp:
                torch.save(model.state_dict(), fp)
