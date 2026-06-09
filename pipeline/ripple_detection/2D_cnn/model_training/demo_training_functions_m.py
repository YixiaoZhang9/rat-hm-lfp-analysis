import os

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


def check_image_count(IMAGE_PATH):
    # avoid having a minibatch of size 1 (normalization issues later on)
    train_yes_path = IMAGE_PATH / "train" / "yes"
    train_no_path = IMAGE_PATH / "train" / "no"
    valid_yes_path = IMAGE_PATH / "val" / "yes"
    valid_no_path = IMAGE_PATH / "val" / "no"

    if len([name for name in os.listdir(train_yes_path) if ".jpg" in name]) % 8 == 1:
        rname = train_yes_path / os.listdir(train_yes_path)[0]
        os.remove(rname)
    if len([name for name in os.listdir(train_no_path) if ".jpg" in name]) % 8 == 1:
        rname = train_no_path / os.listdir(train_no_path)[0]
        os.remove(rname)
    if len([name for name in os.listdir(valid_yes_path) if ".jpg" in name]) % 8 == 1:
        rname = valid_yes_path / os.listdir(valid_yes_path)[0]
        os.remove(rname)
    if len([name for name in os.listdir(valid_no_path) if ".jpg" in name]) % 8 == 1:
        rname = valid_no_path / os.listdir(valid_no_path)[0]
        os.remove(rname)


class TestImageDataset(Dataset):
    # dataset for test images without labels
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        self.fnames = sorted(
            [
                os.path.join(root_dir, fname)
                for fname in os.listdir(root_dir)
                if fname.lower().endswith(valid_exts)
            ]
        )

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img_path = self.fnames[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, img_path


class EarlyStopping:
    # stop training early if validation loss does not improve after patience = 5 iterations
    # load best model
    def __init__(self, patience=5, save_path="best_mod.pth"):
        self.patience = patience
        self.save_path = save_path
        self.best_val_loss = float("inf")
        self.num_epochs_no_improvement = 0
        self.should_stop = False

    def step(self, model, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.num_epochs_no_improvement = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.num_epochs_no_improvement += 1

        if self.num_epochs_no_improvement > self.patience:
            print(f"Stopping - no improvement after {self.patience + 1} epochs")
            self.should_stop = True

    def load_best_model(self, model, device):
        print(f"Loading best model from {self.save_path}")
        state = torch.load(self.save_path, map_location=device)
        model.load_state_dict(state)
        return model


def get_the_data(NEWPATH, arch, batch_size=8):
    # using resnet architecture
    # arch = resnet34
    # define transforms of the data
    sz = 44

    # normalization for pretrained ImageNet models
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # flip vertically to artificially create more training data
    train_tfms = transforms.Compose(
        [
            transforms.Resize((sz, sz)),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=1, scale=(1.0, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )

    valid_tfms = transforms.Compose(
        [
            transforms.Resize((sz, sz)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )

    # get data from path with transforms, batch size 8, test data in 'test' folder
    train_ds = datasets.ImageFolder(
        os.path.join(NEWPATH, "train"), transform=train_tfms
    )
    valid_ds = datasets.ImageFolder(os.path.join(NEWPATH, "val"), transform=valid_tfms)

    test_dir = os.path.join(NEWPATH, "test")
    test_ds = TestImageDataset(test_dir, transform=valid_tfms)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    data = {
        "train_ds": train_ds,
        "val_ds": valid_ds,
        "test_ds": test_ds,
        "train_dl": train_dl,
        "val_dl": valid_dl,
        "test_dl": test_dl,
        "class_names": train_ds.classes,
        "class_to_idx": train_ds.class_to_idx,
        "batch_size": batch_size,
        "image_size": sz,
    }

    return data


def build_model(arch, num_classes):
    # build a pretrained torchvision model and replace the classifier head
    if isinstance(arch, str):
        arch_name = arch.lower()
        if arch_name == "resnet34":
            model = models.resnet34(pretrained=True)
        elif arch_name == "resnet18":
            model = models.resnet18(pretrained=True)
        elif arch_name == "resnet50":
            model = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
    else:
        # allow passing models.resnet34 directly
        model = arch(pretrained=True)

    if not hasattr(model, "fc"):
        raise ValueError("This helper currently expects a torchvision ResNet model.")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model):
    # freeze all layers except the classifier head
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_all(model):
    # unfreeze the whole network
    for param in model.parameters():
        param.requires_grad = True


def run_one_epoch(model, dataloader, criterion, optimizer, device, train=True):
    # run one epoch for training or validation
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    with torch.set_grad_enabled(train):
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            if train:
                loss.backward()
                optimizer.step()

            preds = torch.argmax(logits, dim=1)
            running_loss += loss.item() * xb.size(0)
            running_correct += (preds == yb).sum().item()
            running_total += xb.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    return epoch_loss, epoch_acc


def train_the_model(
    data,
    arch,
    filename,
    old_weight_path=None,
):
    # train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(data["class_names"])

    model = build_model(arch, num_classes)

    # optionally initialize backbone from old fastai weights
    if old_weight_path is not None:
        model = load_old_fastai_backbone_weights(model, old_weight_path, device)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # stage 1: train classifier head
    freeze_backbone(model)
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(5):
        train_loss, train_acc = run_one_epoch(
            model, data["train_dl"], criterion, optimizer, device, train=True
        )
        valid_loss, valid_acc = run_one_epoch(
            model, data["val_dl"], criterion, optimizer, device, train=False
        )
        print(
            f"Stage 1/3 - epoch {epoch + 1}/5 - train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} valid_loss: {valid_loss:.4f} valid_acc: {valid_acc:.4f}"
        )

    # stage 2: unfreeze layer4
    for p in model.layer4.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        [
            {"params": model.layer4.parameters(), "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 5e-4},
        ],
        weight_decay=1e-4,
    )

    for epoch in range(5):
        train_loss, train_acc = run_one_epoch(
            model, data["train_dl"], criterion, optimizer, device, train=True
        )
        valid_loss, valid_acc = run_one_epoch(
            model, data["val_dl"], criterion, optimizer, device, train=False
        )
        print(
            f"Stage 2/3 - epoch {epoch + 1}/5 - train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} valid_loss: {valid_loss:.4f} valid_acc: {valid_acc:.4f}"
        )

    # stage 3: fine-tune the whole network
    unfreeze_all(model)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.conv1.parameters(), "lr": 1e-6},
            {"params": model.bn1.parameters(), "lr": 1e-6},
            {"params": model.layer1.parameters(), "lr": 1e-6},
            {"params": model.layer2.parameters(), "lr": 5e-6},
            {"params": model.layer3.parameters(), "lr": 1e-5},
            {"params": model.layer4.parameters(), "lr": 5e-5},
            {"params": model.fc.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )

    cb = EarlyStopping(save_path="best_mod.pth", patience=3)

    for epoch in range(5):
        train_loss, train_acc = run_one_epoch(
            model, data["train_dl"], criterion, optimizer, device, train=True
        )
        valid_loss, valid_acc = run_one_epoch(
            model, data["val_dl"], criterion, optimizer, device, train=False
        )

        print(
            f"Stage 3/3 - epoch {epoch + 1}/5- train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} valid_loss: {valid_loss:.4f} valid_acc: {valid_acc:.4f}"
        )

        cb.step(model, valid_loss)
        if cb.should_stop:
            break

    model = cb.load_best_model(model, device)

    torch.save(model.state_dict(), filename)
    return model


def test_the_model(data, arch, filename):
    # load in pretrained
    print("Using pretrained model " + filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(data["class_names"])

    model = build_model(arch, num_classes)
    state = torch.load(
        filename, map_location=device
    )  # remove map_location parameter if on GPU
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    class_to_idx = data["class_to_idx"]
    idx_to_class = {value: key for key, value in class_to_idx.items()}

    yes_idx = class_to_idx["Yes"] if "Yes" in class_to_idx else 1

    preds_test = []
    probs_test = []
    test_names = []

    with torch.no_grad():
        for xb, paths in data["test_dl"]:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            preds_test.extend(preds.cpu().numpy().tolist())
            probs_test.extend(probs[:, yes_idx].cpu().numpy().tolist())
            test_names.extend([os.path.basename(p) for p in paths])

    test_df = pd.DataFrame(data=test_names, columns=["image_number"])
    test_df["prediction"] = preds_test
    test_df["probability"] = probs_test
    test_df["predicted_label"] = [idx_to_class[p] for p in preds_test]

    return test_df


def convert_old_fastai_resnet34_state_dict(old_sd):
    # convert old fastai sequential-style keys to torchvision resnet34 keys
    new_sd = {}

    for key, value in old_sd.items():
        if key.startswith("0."):
            new_key = key.replace("0.", "conv1.")
        elif key.startswith("1."):
            new_key = key.replace("1.", "bn1.")
        elif key.startswith("4."):
            new_key = key.replace("4.", "layer1.")
        elif key.startswith("5."):
            new_key = key.replace("5.", "layer2.")
        elif key.startswith("6."):
            new_key = key.replace("6.", "layer3.")
        elif key.startswith("7."):
            new_key = key.replace("7.", "layer4.")
        else:
            # skip old fastai classifier head such as 10, 12, 14, 16
            continue

        new_sd[new_key] = value

    return new_sd


def load_old_fastai_backbone_weights(model, old_weight_path, device):
    # load backbone weights from old fastai model into torchvision resnet34
    old_sd = torch.load(old_weight_path, map_location=device)
    converted_sd = convert_old_fastai_resnet34_state_dict(old_sd)

    model_sd = model.state_dict()

    compatible_sd = {}
    for key, value in converted_sd.items():
        if key in model_sd and model_sd[key].shape == value.shape:
            compatible_sd[key] = value

    model_sd.update(compatible_sd)
    model.load_state_dict(model_sd)

    print(f"Loaded {len(compatible_sd)} backbone tensors from old fastai weights.")
    return model
