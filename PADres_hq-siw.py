from __future__ import print_function

import argparse
import sys
import glob
import os
import random
import math
# Get the current working directory (equivalent to $PWD in bash)
current_working_dir = os.getcwd()
# Add it to the system path
sys.path.append(current_working_dir)
import logging
import csv
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageOps
import cv2  # OpenCV to handle video loading
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


from fvcore.nn import FlopCountAnalysis
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from FIQA.utils import construct_full_model
import torchvision.models as models

def get_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument('--model_name', type=str, default='B_16', choices=[
        "B_16", "B_32", "L_16", "L_32",
        "B_16_imagenet1k", "B_32_imagenet1k", "L_16_imagenet1k", "L_32_imagenet1k"
    ], help='Model architecture to use')
    parser.add_argument('--finetune_method', type=str, default='full', choices=[
        "none", "linear", "full", "partial", "discriminative"
    ], help='Fine-tuning method')
    parser.add_argument('--csv_file_path', type=str, default='PROTOCOL-grand_test-curated.csv', help='Used only in CSV split')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_path', type=str, default='/mnt/c/Users/bouzid/Desktop/FACEPAD3D_project/HQ-WMCA')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--data_augmentation', type=bool, default=False)
    parser.add_argument('--savingtext', type=str, default="")
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.05)')
    
    # Detect if running in notebook
    if "ipykernel" in sys.modules:
        return parser.parse_args([])  # Return default args in notebooks
    else:
        return parser.parse_args()

args = get_args()
weight_decay = args.weight_decay
finetune_method = args.finetune_method
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
gamma = args.gamma
seed = args.seed
data_path = args.data_path
image_size = args.image_size
data_augmentation = args.data_augmentation
savingtext = args.savingtext
csv_file_path = args.csv_file_path

print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("Running on CPU")


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

saving_folder = csv_file_path + "_" + str(lr) + "_" + str(weight_decay) + savingtext

if not os.path.exists("results/"+saving_folder+"/model_weights/"):
    os.makedirs("results/"+saving_folder+"/model_weights/")

log_file = os.path.join("results", saving_folder, "detailed_log"+saving_folder+".txt")
logging.basicConfig(
    filename=log_file, 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("results saved in: " + saving_folder)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything( seed)



csv_file_path = os.path.join(data_path, csv_file_path)  
train_list, val_list, test_list = [], [], []
train_labels, val_labels, test_labels = [], [], []

Impersonation_attacks = os.listdir(os.path.join(data_path, 'HQ-WMCA-RGB', "Impersonation"))

with open(csv_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        vid_name = row[0].split('/')[-1]
        label = int(row[1])
        subset = row[3].lower()
        attack = row[2].split('/')[-1]

        if label == 0:
            full_path = os.path.join(data_path, 'HQ-WMCA-RGB', "Bonafide", vid_name + ".mov")
        else:
            if attack in Impersonation_attacks:
                full_path = glob.glob(os.path.join(data_path, 'HQ-WMCA-RGB', "Impersonation", "*", vid_name + ".mov"))[0]
            else:
                full_path = glob.glob(os.path.join(data_path, 'HQ-WMCA-RGB', "Obfuscation", "*", vid_name + ".mov"))[0]

        label_name = "Bonafide" if label == 0 else "Attack"

        if subset == "train":
            train_list.append(full_path)
            train_labels.append(label_name)
        elif subset == "dev":
            val_list.append(full_path)
            val_labels.append(label_name)
        elif subset == "eval":
            test_list.append(full_path)
            test_labels.append(label_name)


def balance_data(paths, labels):
    bonafide_paths = [p for p, l in zip(paths, labels) if l == "Bonafide"]
    attack_paths   = [p for p, l in zip(paths, labels) if l == "Attack"]

    n_bonafide = len(bonafide_paths)
    n_attack   = len(attack_paths)

    if n_bonafide < n_attack:
        repeat_factor = math.ceil(n_attack / n_bonafide)
        bonafide_paths = (bonafide_paths * repeat_factor)[:n_attack]
    elif n_attack < n_bonafide:
        repeat_factor = math.ceil(n_bonafide / n_attack)
        attack_paths = (attack_paths * repeat_factor)[:n_bonafide]

    # Merge and shuffle
    all_paths = bonafide_paths + attack_paths
    all_labels = ["Bonafide"] * len(bonafide_paths) + ["Attack"] * len(attack_paths)
    combined = list(zip(all_paths, all_labels))
    random.shuffle(combined)
    paths, labels = zip(*combined)
    return list(paths), list(labels)

train_list, train_labels = balance_data(train_list+val_list, train_labels+val_labels)
val_list, val_labels = balance_data(test_list, test_labels)
    
logging.info(f"Nombre d'exemples d'entraînement: {len(train_list)}")
logging.info(f"Nombre d'exemples de validation: {len(val_list)}")


class HQ_WMCA_RGB(Dataset):
    def __init__(self, video_paths, labels, transform=None, fiqa_transform=None, target_size=(224, 224)):
        """
        Args:
            video_paths (list): List of paths to video files.
            labels (list): List of labels corresponding to each video.
            transform (callable, optional): Optional transform to be applied on each frame.
            target_size (tuple): The size to which each frame will be resized (default is 224x224 for ViT).
        """
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.fiqa_transform = fiqa_transform
        self.target_size = target_size
        self.data = []  # List to hold all frames and their labels

        # Process each video and store each frame
        for video_path, x in tqdm(zip(video_paths, labels), total=len(video_paths), desc="Processing videos"):
            # print(f"Processing video: {video_path}","label",label)
            frames = self._load_video(video_path)
            for i, frame in enumerate(frames):
                if x == "Bonafide":
                    label = 0.0
                else:
                    label = 1.0
                generated = Image.open(video_path
                   .replace("HQ-WMCA-RGB", "HQ-WMCA_e4e")
                   .replace(".mov", "/img_"+str(i)+".jpg")).convert("RGB").copy()
                generated = generated.resize(self.target_size)
                # print(label)
                self.data.append((frame, generated, label))  # Store each frame with its label
            
                # print(self.data)

    def _load_video(self, video_path):
        """
        Load a video and extract frames as images.
        Args:
            video_path (str): Path to the video file.
        
        Returns:
            List of frames (PIL images).
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert frame from BGR (OpenCV) to RGB (PIL)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            frame = Image.fromarray(frame)
            # Resize frame to target size
            frame = frame.resize(self.target_size)
            frames.append(frame)
        
        cap.release()
        return frames

    def __len__(self):
        """Return the total number of frames (data points)."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return the frame and its corresponding label at the specified index."""
        frame, generated, label = self.data[idx]
        
        # Random horizontal flip (consistent for both frame & generated)
        if torch.rand(1).item() > 0.5:
            frame = ImageOps.mirror(frame)
            generated = ImageOps.mirror(generated)

        # Apply transforms
        pad_frame = self.transform(frame).float() if self.transform else frame
        fiqa_pad_frame = self.fiqa_transform(frame).float() if self.fiqa_transform else frame
        fiqa_generated = self.fiqa_transform(generated).float() if self.fiqa_transform else generated

        return pad_frame, fiqa_pad_frame, fiqa_generated, label



# Normalization values
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

fiqa_mean = [0.5, 0.5, 0.5]
fiqa_std  = [0.5, 0.5, 0.5]

# ---------- Train transforms ----------
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),       
    transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                           saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

fiqa_train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(112, scale=(0.85, 1.0)),       
    transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                           saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=fiqa_mean, std=fiqa_std),  # scale to [-1,1]
])

# ---------- Validation transforms ----------
val_transforms = transforms.Compose([
    transforms.Resize(224),              # resize shorter side
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

fiqa_val_transforms = transforms.Compose([
    transforms.Resize(112),
    transforms.CenterCrop(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=fiqa_mean, std=fiqa_std),
])

# train_list = train_list[::800]
# train_labels = train_labels[::800]
# val_list = val_list[::800]
# val_labels = val_labels[::800]
# test_list = test_list[::800]
# test_labels = test_labels[::800]



train_data = HQ_WMCA_RGB(train_list, train_labels, transform=train_transforms, fiqa_transform=fiqa_train_transforms, target_size= (image_size, image_size))
valid_data = HQ_WMCA_RGB(val_list, val_labels, transform=val_transforms, fiqa_transform=fiqa_val_transforms, target_size= (image_size, image_size))
test_data = HQ_WMCA_RGB(test_list, test_labels, transform=val_transforms, fiqa_transform=fiqa_val_transforms, target_size= (image_size, image_size))



train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=8)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=8)

print("train_data", len(train_data),"train_loader", len(train_loader))
print("valid_data", len(valid_data),"valid_loader", len(valid_loader))
print("test_data", len(test_data),"test_loader", len(test_loader))


class ResNet50WithGenerated(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        # ------- Independent ResNet50 backbones (pre-trained) -------
        backbone_rgb = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove final fc layers (keep feature extractor)
        self.resnet = nn.Sequential(*list(backbone_rgb.children())[:-1])      # for RGB input
        self.fiqa_model, _, _ = construct_full_model("FIQA/configs/model_config.yaml")
        self.fiqa_model.load_state_dict(torch.load("FIQA/aikd_diffiqar_model.pth"))

        self.resnet_feat_dim = 2048  # resnet50 output feature dim
        self.fiqa_feat_dim = 512  # resnet50 output feature dim


        self.resnet_proj = nn.Linear( self.resnet_feat_dim, self.fiqa_feat_dim)
        self.fiqa_proj = nn.Linear( self.fiqa_feat_dim, self.fiqa_feat_dim)


        # MLP heads
        self.mlp_resnet = nn.Sequential(
            nn.LayerNorm(self.fiqa_feat_dim),
            nn.Dropout(p=dropout),
            nn.Linear(self.fiqa_feat_dim, self.fiqa_feat_dim // 2),
            nn.ReLU(),
            nn.Linear(self.fiqa_feat_dim // 2, num_classes)
        )

        self.mlp_gen = nn.Sequential(
            nn.LayerNorm(self.fiqa_feat_dim),
            nn.Dropout(p=dropout),
            nn.Linear(self.fiqa_feat_dim, self.fiqa_feat_dim // 2),
            nn.ReLU(),
            nn.Linear(self.fiqa_feat_dim // 2, num_classes)
        )

        self.mlp_glob = nn.Sequential(
            nn.LayerNorm(2 * self.fiqa_feat_dim),
            nn.Dropout(p=dropout),
            nn.Linear(2*self.fiqa_feat_dim, self.fiqa_feat_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.fiqa_feat_dim, num_classes)
        )
        self.activation = nn.Sigmoid()

    # ------------------------------------------------------------
    def forward(self, image: torch.Tensor, image_112: torch.Tensor, imageGen: torch.Tensor):
        """
        image:     [batch, 3, H, W]   (e.g. 224×224)
        imageGen:  [batch, 3, H, W]
        """
        # Forward through *independent* ResNet50 backbones
        rgb_feat = self.resnet(image)        # [B, 2048, 1, 1]
        with torch.no_grad():
            rgbfiqa_feat, _ = self.fiqa_model(image_112)  # [B, 512]
            genfiqa_feat, _ = self.fiqa_model(imageGen)  # [B, 512]

        rgb_feat = rgb_feat.view(rgb_feat.size(0), -1)
        rgb_feat = self.resnet_proj(rgb_feat)
        feats = rgbfiqa_feat - genfiqa_feat
        feats = self.fiqa_proj(feats)

        fused = torch.cat([rgb_feat, feats], dim=1) # [B, D]

        logits_resnet = self.mlp_resnet(rgb_feat)
        logits_gen = self.mlp_gen(feats)

        logits = self.mlp_glob(fused)
        probs = self.activation(logits)
        probs_resnet = self.activation(logits_resnet)
        probs_gen = self.activation(logits_gen)

        return probs, probs_resnet, probs_gen, fused

model = ResNet50WithGenerated(num_classes=1)

model = model.to(device)


# Print the model details
num_params = count_parameters(model)
print(num_params, "M param")

# Create a dummy input of the correct shape
dummy_input = torch.randn(4, 3, image_size, image_size).to(device)
dummy_input2 = torch.randn(4, 3, 112, 112).to(device)

# Model info
logging.info(f"Model type: {type(model)}")
logging.info(f"Number of parameters: {num_params} M")
logging.info(f"Batch size: {batch_size}")
logging.info(f"Image size: {image_size}")
logging.info(f"Data augmentation: {data_augmentation}")
logging.info(f"Training epochs: {epochs}")
logging.info(f"Learning rate: {lr}")
logging.info(f"Gamma: {gamma}")
logging.info(f"Seed: {seed}")
logging.info(f"Fine-tuning method: {finetune_method}")
# Calculate FLOPs
flops = FlopCountAnalysis(model, (dummy_input, dummy_input2, dummy_input2))
logging.info(f"FLOPs: {flops.total()}")
print("Model name:")
print("Model type:")
print(type(model))


# Count the frequency of each class label
label_counts = Counter()
for _, _,_, labels in train_loader:
    # Assuming labels are NOT one-hot, shape: [batch_size]
    label_counts.update(labels.tolist())

# Total samples and number of classes
num_classes = len(label_counts)
total = sum(label_counts.values())


print(num_classes)
print(label_counts)

class OCCL(torch.nn.Module):
    def __init__(self, margin=3.0, feat_dim=10, alpha=0.1, device=None, center_adapt=True):
        super(OCCL, self).__init__()
        self.device = device
        self.margin = margin
        self.feat_dim = feat_dim
        self.center_adapt = center_adapt
        if self.center_adapt:
            self.center = torch.randn(1, self.feat_dim, requires_grad=False).to(self.device)
        else:
            self.center = torch.zeros(1, self.feat_dim, requires_grad=False).to(self.device)

        self.alpha = alpha

    def forward(self, x, label, is_training=True):
        """
        x: tensor of shape [batch_size, feat_dim]
        label: tensor of shape [batch_size], values in {0, 1}
        """
        batch_size = x.size(0)

        # Use boolean mask instead of numpy
        bonafide_mask = (label == 0).squeeze(-1)

        if is_training and self.center_adapt and bonafide_mask.any():
            bonafide_features = x[bonafide_mask]
            self.center = self.center + self.alpha * torch.mean(
                bonafide_features.detach() - self.center.expand_as(bonafide_features),
                dim=0, keepdim=True
            )

        expanded_centers = self.center.expand(batch_size, -1)
        euclidean_distance = F.pairwise_distance(x, expanded_centers)

        # class 0 = bonafide → pull to center
        # class 1 = attack → push away (margin)
        bonafide_loss = (1 - label) * torch.pow(euclidean_distance, 2)
        attack_loss = label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)

        loss = torch.mean(bonafide_loss + attack_loss)
        return loss
    

criterion = nn.BCELoss()
occl_loss_fn = OCCL(margin=10.0, feat_dim=2*model.fiqa_feat_dim, alpha=0.1, device=device, center_adapt=True)


# Fine-tune the entire model
for param in model.parameters():
    param.requires_grad = True
for param in model.fiqa_model.parameters():
    param.requires_grad = False


num_finetune_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters to be trained: {num_finetune_params/1000000} M")


# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# scheduler
scheduler = StepLR(optimizer, step_size=2, gamma=gamma)

logging.info(f"Optimizer: {optimizer}")
logging.info(f"Scheduler: {scheduler}")
logging.info(f"Criterion: {criterion}")

best_val_acc = 0.0

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

logging.info(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
# Training loop

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0

    for data, data_fiqa, gen_data, label in tqdm(train_loader):
        data = data.to(device)
        data_fiqa = data_fiqa.to(device)
        gen_data = gen_data.to(device)
        label = label.to(device)

        probs, probs_vit, probs_gen, feats = model(data, data_fiqa, gen_data)
        label = label.float().unsqueeze(1)

        loss = criterion(probs, label) + 0.001*occl_loss_fn(feats, label, is_training=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = ((probs >= 0.5).float() == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

        #memory=torch.cuda.max_memory_allocated() / MB
        #print(memory)
    scheduler.step()
    print("memory usage ",torch.cuda.max_memory_allocated() / 1e9, "GB")

    model.eval()
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, data_fiqa, gen_data, label in valid_loader:
            data = data.to(device)
            data_fiqa = data_fiqa.to(device)
            gen_data = gen_data.to(device)
            label = label.to(device)
            label = label.float().unsqueeze(1)
            probs, probs_vit, probs_gen, feats = model(data, data_fiqa, gen_data)
            val_loss = criterion(probs, label) 

            acc = ((probs >= 0.5).float() == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )
    logging.info(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )
    train_losses.append(epoch_loss)
    val_losses.append(epoch_val_loss)
    train_accuracies.append(epoch_accuracy)
    val_accuracies.append(epoch_val_accuracy)

    
    if epoch_val_accuracy>=best_val_acc:
        best_val_acc = epoch_val_accuracy
        torch.save(model.state_dict(), os.path.join('results',saving_folder,'model_weights','model.ckpt'))
    torch.cuda.empty_cache()

logging.info(f"Training ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")


train_losses = [float(loss) for loss in train_losses]
val_losses = [float(loss) for loss in val_losses]
train_accuracies = [float(accc) for accc in train_accuracies]
val_accuracies = [float(accc) for accc in val_accuracies]

# Plot and save loss curve
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join("results", saving_folder, "loss_plot"+saving_folder+".png"))
plt.close()  # Close the figure to free memory

# Plot and save accuracy curve
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy', marker='o')
plt.plot(val_accuracies, label='Validation Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join("results", saving_folder, "accuracy_plot"+saving_folder+".png"))
plt.close()


print('This is results for the best model.')
print("####################################")

# === Load model ===
model.load_state_dict(torch.load(os.path.join('results', saving_folder, 'model_weights', 'model.ckpt')))
model.eval()

# =========================
# Step 1: Compute validation scores (for EER threshold)
# =========================
val_labels_list = []
val_scores_list = []

with torch.no_grad():
    for data, data_fiqa, gen_data, label in valid_loader:
        data, data_fiqa, gen_data, label = data.to(device), data_fiqa.to(device), gen_data.to(device), label.to(device)
        probs, probs_vit, probs_gen, feats = model(data, data_fiqa, gen_data)
        val_labels_list.extend(label.cpu().numpy())
        val_scores_list.extend(probs.cpu().numpy())

val_labels_array = np.array(val_labels_list)
val_scores_array = np.array(val_scores_list)

# Compute validation ROC & EER
fpr_val, tpr_val, thresholds_val = roc_curve(val_labels_array, val_scores_array)
eer_idx = np.argmin(np.abs(fpr_val - (1 - tpr_val)))
eer_threshold = thresholds_val[eer_idx]


del train_data, train_labels, train_list, train_loader
del valid_data, val_labels, val_list, valid_loader

siw_list = []
siw_labels = []

live_paths = glob.glob(os.path.join(data_path.replace('HQ-WMCA','siw-mv2'), "SiW-Mv2-crop", "Live", "*"))

for full_path in live_paths:
    siw_list.append(full_path)
    siw_labels.append("Bonafide")

attack_paths = glob.glob(os.path.join(data_path.replace('HQ-WMCA','siw-mv2'), 'SiW-Mv2-crop', "Spoof", "*", "*"))

for attack_path in attack_paths:
    siw_list.append(attack_path)
    siw_labels.append("Attack")


class Siwmv2(Dataset):
    def __init__(self, video_paths, labels, transform=None, fiqa_transform=None, target_size=(224, 224), num_samples=100):
        self.video_paths = video_paths          # paths to crop folders
        self.labels = labels
        self.transform = transform
        self.fiqa_transform = fiqa_transform
        self.target_size = target_size
        self.num_samples = num_samples
        self.data = []

        for rgb_folder, x in tqdm(zip(video_paths, labels), total=len(video_paths), desc="Processing vids"):
            # Define the corresponding e4e folder
            gen_folder = rgb_folder.replace("SiW-Mv2-crop", "SiW-Mv2-e4e").split(".")[0]

            # Check if both folders exist
            if not os.path.isdir(rgb_folder) or not os.path.isdir(gen_folder):
                continue

            # Count frames in the crop folder (reference)
            existing_frames = sorted([f for f in os.listdir(rgb_folder) if f.endswith(".jpg")])
            n_existing = len(existing_frames)
            if n_existing == 0:
                continue

            # Choose indices
            if n_existing >= self.num_samples:
                indices = np.linspace(0, n_existing - 1, self.num_samples, dtype=int)
            else:
                repeat_times = int(np.ceil(self.num_samples / n_existing))
                indices = np.tile(np.arange(n_existing), repeat_times)[:self.num_samples]

            # Build image paths
            rgb_paths = [os.path.join(rgb_folder, f"img_{i}.jpg") for i in indices]
            gen_paths = [os.path.join(gen_folder, f"img_{i}.jpg") for i in indices]

            # Load both sets of images
            rgb_frames = self._load_images_parallel(rgb_paths)
            generated_frames = self._load_images_parallel(gen_paths)

            label = 0.0 if x == "Bonafide" else 1.0
            # print(f"RAM after {os.path.basename(rgb_folder)}: {ram_usage:.2f} GB")

            # Append tuples (rgb generated, label)
            for rgb, generated in zip(rgb_frames, generated_frames):
                if rgb is not None and generated is not None:
                    self.data.append((rgb, generated, label))

            # free intermediate memory
            rgb_frames.clear()
            generated_frames.clear()

    def _load_images_parallel(self, paths):
        """Load images concurrently, safely."""
        def safe_load(p):
            with Image.open(p) as im:
                return im.convert("RGB").resize(self.target_size)


        with ThreadPoolExecutor(max_workers=8) as executor:
            return list(executor.map(safe_load, paths))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb, generated, label = self.data[idx]

        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            rgb = ImageOps.mirror(rgb)
            generated = ImageOps.mirror(generated)

        # Apply transforms
        rgb_tensor = self.transform(rgb).float()
        fiqa_tensor = self.fiqa_transform(rgb).float()
        gen_tensor = self.fiqa_transform(generated).float()

        # Return crop, generated, and label
        return rgb_tensor, fiqa_tensor, gen_tensor, label
    

class HQ_WMCA_RGB(Dataset):
    def __init__(self, video_paths, labels, transform=None, fiqa_transform=None, target_size=(224, 224)):
        """
        Args:
            video_paths (list): List of paths to video files.
            labels (list): List of labels corresponding to each video.
            transform (callable, optional): Optional transform to be applied on each frame.
            target_size (tuple): The size to which each frame will be resized (default is 224x224 for ViT).
        """
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.fiqa_transform = fiqa_transform
        self.target_size = target_size
        self.data = []  # List to hold all frames and their labels

        # Process each video and store each frame
        for video_path, x in tqdm(zip(video_paths, labels), total=len(video_paths), desc="Processing videos"):
            # print(f"Processing video: {video_path}","label",label)
            frames = self._load_video(video_path)
            for i, frame in enumerate(frames):
                if x == "Bonafide":
                    label = 0.0
                else:
                    label = 1.0
                generated = Image.open(video_path
                   .replace("HQ-WMCA-RGB", "HQ-WMCA_e4e")
                   .replace(".mov", "/img_"+str(i)+".jpg")).convert("RGB").copy()
                generated = generated.resize(self.target_size)
                # print(label)
                self.data.append((frame, generated, label))  # Store each frame with its label
            
                # print(self.data)

    def _load_video(self, video_path):
        """
        Load a video and extract frames as images.
        Args:
            video_path (str): Path to the video file.
        
        Returns:
            List of frames (PIL images).
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert frame from BGR (OpenCV) to RGB (PIL)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            frame = Image.fromarray(frame)
            # Resize frame to target size
            frame = frame.resize(self.target_size)
            frames.append(frame)
        
        cap.release()
        return frames

    def __len__(self):
        """Return the total number of frames (data points)."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return the frame and its corresponding label at the specified index."""
        frame, generated, label = self.data[idx]
        
        # Random horizontal flip (consistent for both frame & generated)
        if torch.rand(1).item() > 0.5:
            frame = ImageOps.mirror(frame)
            generated = ImageOps.mirror(generated)

        # Apply transforms
        pad_frame = self.transform(frame).float() if self.transform else frame
        fiqa_pad_frame = self.fiqa_transform(frame).float() if self.fiqa_transform else frame
        fiqa_generated = self.fiqa_transform(generated).float() if self.fiqa_transform else generated

        return pad_frame, fiqa_pad_frame, fiqa_generated, label


test_list = siw_list
test_labels = siw_labels

# test_list = test_list[::800]
# test_labels = test_labels[::800]

test_data = Siwmv2(test_list, test_labels, transform=val_transforms, fiqa_transform=fiqa_val_transforms, target_size= (image_size, image_size))
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=8)
print("test_data", len(test_data),"test_loader", len(test_loader))


def compute_eer(fpr, tpr):
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    return fpr[min_index]

def compute_apcer(y_true, y_pred):
    attack_mask = (y_true == 1)
    if np.sum(attack_mask) == 0:
        return 0.0
    return np.sum((y_pred == 0) & attack_mask) / np.sum(attack_mask)

def compute_bpcer(y_true, y_pred):
    bona_mask = (y_true == 0)
    if np.sum(bona_mask) == 0:
        return 0.0
    return np.sum((y_pred == 1) & bona_mask) / np.sum(bona_mask)

test_list = []

with torch.no_grad():
    total_test_accuracy = 0
    total_test_loss = 0
    for data, data_fiqa, gen_data, label in test_loader:
        data = data.to(device)
        data_fiqa = data_fiqa.to(device)
        gen_data = gen_data.to(device)
        label = label.to(device)
        label = label.float().unsqueeze(1)
        probs, probs_vit, probs_gen, feats = model(data, data_fiqa, gen_data)
        test_loss = criterion(probs, label)
        test_list.append([label.cpu(), probs.cpu()])

        acc = ((probs >= 0.5).float() == label).float().mean()
        total_test_accuracy += acc / len(test_loader)
        total_test_loss += test_loss / len(test_loader)

print(f"\nTest results - loss: {total_test_loss:.4f} - acc: {total_test_accuracy:.4f}")
logging.info(f"\nTest results - loss: {total_test_loss:.4f} - acc: {total_test_accuracy:.4f}")
# =======================
# Compute Extra Metrics
# =======================

# Prepare labels and predictions
y_true = []
y_pred = []
y_scores = []

for label, output in test_list:
    true = label.numpy()  # one-hot to index
    pred = (output >= 0.5).float()

    y_true.extend(true)
    y_pred.extend(pred)
    y_scores.extend(pred)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_scores = np.array(y_scores)


# Binary classification: 0 = bona fide, 1 = attack
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)

# ROC & AUC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
auc = roc_auc_score(y_true, y_scores)


apcer = compute_apcer(y_true, y_pred)
bpcer = compute_bpcer(y_true, y_pred)
acer = (apcer + bpcer) / 2

# HTER (at EER threshold)
eer_threshold_index = np.argmin(np.abs(fpr - (1 - tpr)))
threshold_at_eer = thresholds[eer_threshold_index]
y_pred_eer = (y_scores >= threshold_at_eer).astype(int)
far = fpr[eer_threshold_index]
frr = 1 - tpr[eer_threshold_index]
hter = (far + frr) / 2

# ========================
# Print all metrics
# ========================
print(f"""
Detailed PAD Evaluation:
----------------------------
Accuracy : {accuracy:.4f}
Precision: {precision:.4f}
Recall   : {recall:.4f}
AUC      : {auc:.4f}
HTER     : {hter:.4f}
APCER    : {apcer:.4f}
BPCER    : {bpcer:.4f}
ACER     : {acer:.4f}
""")
logging.info(f"""
Detailed PAD Evaluation:
----------------------------
Accuracy : {accuracy:.4f}
Precision: {precision:.4f}
Recall   : {recall:.4f}
AUC      : {auc:.4f}
HTER     : {hter:.4f}
APCER    : {apcer:.4f}
BPCER    : {bpcer:.4f}
ACER     : {acer:.4f}
""")

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
labels = ['Bona Fide', 'Attack']


# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # or use labels if it's not binary
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

# Plot
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.tight_layout()

# Save to file
plt.savefig(os.path.join("results", saving_folder, f"confusion_matrix_{saving_folder}.png"))
plt.show()


# Group outputs by video
video_dict = {}  # video_id -> list of (true_label, score, pred)
for idx in range(len(y_true)):
    video_id = idx // 60  # assumes exactly 60 frames per video
    if video_id not in video_dict:
        video_dict[video_id] = []
    video_dict[video_id].append((y_true[idx], y_scores[idx], y_pred[idx]))

# Function to aggregate predictions from 10 frames
def aggregate_frames(frames, strategy="mean"):
    true_label = frames[0][0]  # same label for all frames in video
    scores = np.array([f[1] for f in frames])
    preds = np.array([f[2] for f in frames])

    if strategy == "mean":
        # take mean score of 10 frames
        final_score = np.mean(scores)
    elif strategy == "most_certain":
        # certainty = distance from 0.5
        certainty = np.abs(scores - 0.5)
        top_idx = np.argsort(-certainty)[:10]  # top 10 certain
        final_score = np.mean(scores[top_idx])
    else:
        raise ValueError("Unknown strategy")

    final_pred = int(final_score >= 0.5)  # threshold 0.5
    return true_label, final_pred, final_score

# Collect video-level results
y_true, y_pred, y_scores = [], [], []
for vid, frames in video_dict.items():
    # pick 10 evenly spaced frames
    step = len(frames) // 10
    selected_frames = frames[::step][:10]

    # 1) mean strategy
    true, pred, score = aggregate_frames(selected_frames, strategy="mean")
    y_true.append(true)
    y_pred.append(pred)
    y_scores.append(score)

    # 2) alternatively: certainty strategy
    # true, pred, score = aggregate_frames(frames, strategy="most_certain")

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_scores = np.array(y_scores)

# ==== same evaluation pipeline ====
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
auc = roc_auc_score(y_true, y_scores)

# APCER, BPCER, ACER
apcer = compute_apcer(y_true, y_pred)
bpcer = compute_bpcer(y_true, y_pred)
acer = (apcer + bpcer) / 2

print(f"""
Video-level PAD Evaluation (10 frames, unif):
---------------------------------------
Accuracy : {accuracy:.4f}
Precision: {precision:.4f}
Recall   : {recall:.4f}
AUC      : {auc:.4f}
APCER    : {apcer:.4f}
BPCER    : {bpcer:.4f}
ACER     : {acer:.4f}
""")


logging.info(f"""
Video-level PAD Evaluation (10 frames, unif):
---------------------------------------
Accuracy : {accuracy:.4f}
Precision: {precision:.4f}
Recall   : {recall:.4f}
AUC      : {auc:.4f}
APCER    : {apcer:.4f}
BPCER    : {bpcer:.4f}
ACER     : {acer:.4f}
""")



print(f"Validation EER threshold: {eer_threshold:.4f}")
logging.info(f"############## Validation EER threshold: {eer_threshold:.4f} ###################")
# =========================
# Step 2: Compute test scores
# =========================
test_labels_list = []
test_scores_list = []
test_preds_list = []

with torch.no_grad():
    total_test_loss = 0
    total_test_acc = 0
    for data, data_fiqa, gen_data, label in test_loader:
        data, data_fiqa, gen_data, label = data.to(device), data_fiqa.to(device), gen_data.to(device), label.to(device)
        probs, probs_vit, probs_gen, feats = model(data, data_fiqa, gen_data)

        # save for metrics
        test_labels_list.extend(label.cpu().numpy())
        test_scores_list.extend(probs.cpu().numpy())
        test_preds_list.extend((probs >= eer_threshold).cpu().numpy())  # Apply validation EER threshold
        
        label = label.float().unsqueeze(1)
        acc = ((probs >= 0.5).float() == label).float().mean()
        total_test_acc += acc / len(test_loader)
        test_loss = criterion(probs, label) 
        total_test_loss += test_loss / len(test_loader)

y_true = np.array(test_labels_list)
y_scores = np.array(test_scores_list)
y_pred = np.array(test_preds_list)

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
auc = roc_auc_score(y_true, y_scores)

# APCER / BPCER / ACER
apcer = np.mean(y_pred[y_true == 1] != 1)  # False negatives for attacks
bpcer = np.mean(y_pred[y_true == 0] != 0)  # False positives for bona fide
acer = (apcer + bpcer) / 2

# HTER at EER threshold
fpr_test, tpr_test, thresholds_test = roc_curve(y_true, y_scores)
far = np.mean(y_scores[y_true == 0] >= eer_threshold)
frr = np.mean(y_scores[y_true == 1] < eer_threshold)
hter = (far + frr) / 2

print(f"""
Test results:
-------------
Loss      : {total_test_loss:.4f}
Accuracy  : {accuracy:.4f}
Precision : {precision:.4f}
Recall    : {recall:.4f}
AUC       : {auc:.4f}
HTER      : {hter:.4f}
APCER     : {apcer:.4f}
BPCER     : {bpcer:.4f}
ACER      : {acer:.4f}
""")
logging.info(f"""
Test results:
-------------
Loss      : {total_test_loss:.4f}
Accuracy  : {accuracy:.4f}
Precision : {precision:.4f}
Recall    : {recall:.4f}
AUC       : {auc:.4f}
HTER      : {hter:.4f}
APCER     : {apcer:.4f}
BPCER     : {bpcer:.4f}
ACER      : {acer:.4f}
""")


# =========================
# Step 3: Video-level evaluation
# =========================
video_dict = {}
frames_per_video = 100
for idx in range(len(y_true)):
    vid = idx // frames_per_video
    if vid not in video_dict:
        video_dict[vid] = []
    video_dict[vid].append((y_true[idx], y_scores[idx], y_pred[idx]))

def aggregate_frames(frames, strategy="mean"):
    true_label = frames[0][0]
    scores = np.array([f[1] for f in frames])
    if strategy == "mean":
        final_score = np.mean(scores)
    elif strategy == "most_certain":
        certainty = np.abs(scores - 0.5)
        top_idx = np.argsort(-certainty)[:20]
        final_score = np.mean(scores[top_idx])
    else:
        raise ValueError("Unknown strategy")
    final_pred = int(final_score >= eer_threshold)
    return true_label, final_pred, final_score

y_true_vid, y_pred_vid, y_scores_vid = [], [], []
for vid, frames in video_dict.items():
    true, pred, score = aggregate_frames(frames, strategy="mean")
    y_true_vid.append(true)
    y_pred_vid.append(pred)
    y_scores_vid.append(score)

y_true_vid = np.array(y_true_vid)
y_pred_vid = np.array(y_pred_vid)
y_scores_vid = np.array(y_scores_vid)

accuracy_vid = accuracy_score(y_true_vid, y_pred_vid)
precision_vid = precision_score(y_true_vid, y_pred_vid, zero_division=0)
recall_vid = recall_score(y_true_vid, y_pred_vid, zero_division=0)
auc_vid = roc_auc_score(y_true_vid, y_scores_vid)

apcer_vid = np.mean(y_pred_vid[y_true_vid == 1] != 1)
bpcer_vid = np.mean(y_pred_vid[y_true_vid == 0] != 0)
acer_vid = (apcer_vid + bpcer_vid) / 2

far_vid = np.mean(y_scores_vid[y_true_vid == 0] >= eer_threshold)
frr_vid = np.mean(y_scores_vid[y_true_vid == 1] < eer_threshold)
hter_vid = (far_vid + frr_vid) / 2

print(f"""
Full Video-level PAD Evaluation:
---------------------------
Accuracy  : {accuracy_vid:.4f}
Precision : {precision_vid:.4f}
Recall    : {recall_vid:.4f}
AUC       : {auc_vid:.4f}
HTER      : {hter_vid:.4f}
APCER     : {apcer_vid:.4f}
BPCER     : {bpcer_vid:.4f}
ACER      : {acer_vid:.4f}
""")
logging.info(f"""
Full Video-level PAD Evaluation:
---------------------------
Accuracy  : {accuracy_vid:.4f}
Precision : {precision_vid:.4f}
Recall    : {recall_vid:.4f}
AUC       : {auc_vid:.4f}
HTER      : {hter_vid:.4f}
APCER     : {apcer_vid:.4f}
BPCER     : {bpcer_vid:.4f}
ACER      : {acer_vid:.4f}
""")


def aggregate_frames(frames, strategy="mean"):
    true_label = frames[0][0]
    scores = np.array([f[1] for f in frames])
    if strategy == "mean":
        final_score = np.mean(scores[::5])
    elif strategy == "most_certain":
        certainty = np.abs(scores - 0.5)
        top_idx = np.argsort(-certainty)[:20]
        final_score = np.mean(scores[top_idx])
    else:
        raise ValueError("Unknown strategy")
    final_pred = int(final_score >= eer_threshold)
    return true_label, final_pred, final_score

y_true_vid, y_pred_vid, y_scores_vid = [], [], []
for vid, frames in video_dict.items():
    true, pred, score = aggregate_frames(frames, strategy="mean")
    y_true_vid.append(true)
    y_pred_vid.append(pred)
    y_scores_vid.append(score)

y_true_vid = np.array(y_true_vid)
y_pred_vid = np.array(y_pred_vid)
y_scores_vid = np.array(y_scores_vid)

accuracy_vid = accuracy_score(y_true_vid, y_pred_vid)
precision_vid = precision_score(y_true_vid, y_pred_vid, zero_division=0)
recall_vid = recall_score(y_true_vid, y_pred_vid, zero_division=0)
auc_vid = roc_auc_score(y_true_vid, y_scores_vid)

apcer_vid = np.mean(y_pred_vid[y_true_vid == 1] != 1)
bpcer_vid = np.mean(y_pred_vid[y_true_vid == 0] != 0)
acer_vid = (apcer_vid + bpcer_vid) / 2

far_vid = np.mean(y_scores_vid[y_true_vid == 0] >= eer_threshold)
frr_vid = np.mean(y_scores_vid[y_true_vid == 1] < eer_threshold)
hter_vid = (far_vid + frr_vid) / 2

print(f"""
20 Frames Video-level PAD Evaluation:
---------------------------
Accuracy  : {accuracy_vid:.4f}
Precision : {precision_vid:.4f}
Recall    : {recall_vid:.4f}
AUC       : {auc_vid:.4f}
HTER      : {hter_vid:.4f}
APCER     : {apcer_vid:.4f}
BPCER     : {bpcer_vid:.4f}
ACER      : {acer_vid:.4f}
""")
logging.info(f"""
20 Frames Video-level PAD Evaluation:
---------------------------
Accuracy  : {accuracy_vid:.4f}
Precision : {precision_vid:.4f}
Recall    : {recall_vid:.4f}
AUC       : {auc_vid:.4f}
HTER      : {hter_vid:.4f}
APCER     : {apcer_vid:.4f}
BPCER     : {bpcer_vid:.4f}
ACER      : {acer_vid:.4f}
""")

def aggregate_frames(frames, strategy="mean"):
    true_label = frames[0][0]
    scores = np.array([f[1] for f in frames])
    if strategy == "mean":
        final_score = np.mean(scores[::10])
    elif strategy == "most_certain":
        certainty = np.abs(scores - 0.5)
        top_idx = np.argsort(-certainty)[:10]
        final_score = np.mean(scores[top_idx])
    else:
        raise ValueError("Unknown strategy")
    final_pred = int(final_score >= eer_threshold)
    return true_label, final_pred, final_score

y_true_vid, y_pred_vid, y_scores_vid = [], [], []
for vid, frames in video_dict.items():
    true, pred, score = aggregate_frames(frames, strategy="mean")
    y_true_vid.append(true)
    y_pred_vid.append(pred)
    y_scores_vid.append(score)

y_true_vid = np.array(y_true_vid)
y_pred_vid = np.array(y_pred_vid)
y_scores_vid = np.array(y_scores_vid)

accuracy_vid = accuracy_score(y_true_vid, y_pred_vid)
precision_vid = precision_score(y_true_vid, y_pred_vid, zero_division=0)
recall_vid = recall_score(y_true_vid, y_pred_vid, zero_division=0)
auc_vid = roc_auc_score(y_true_vid, y_scores_vid)

apcer_vid = np.mean(y_pred_vid[y_true_vid == 1] != 1)
bpcer_vid = np.mean(y_pred_vid[y_true_vid == 0] != 0)
acer_vid = (apcer_vid + bpcer_vid) / 2

far_vid = np.mean(y_scores_vid[y_true_vid == 0] >= eer_threshold)
frr_vid = np.mean(y_scores_vid[y_true_vid == 1] < eer_threshold)
hter_vid = (far_vid + frr_vid) / 2

print(f"""
10 Frames Video-level PAD Evaluation:
---------------------------
Accuracy  : {accuracy_vid:.4f}
Precision : {precision_vid:.4f}
Recall    : {recall_vid:.4f}
AUC       : {auc_vid:.4f}
HTER      : {hter_vid:.4f}
APCER     : {apcer_vid:.4f}
BPCER     : {bpcer_vid:.4f}
ACER      : {acer_vid:.4f}
""")
logging.info(f"""
10 Frames Video-level PAD Evaluation:
---------------------------
Accuracy  : {accuracy_vid:.4f}
Precision : {precision_vid:.4f}
Recall    : {recall_vid:.4f}
AUC       : {auc_vid:.4f}
HTER      : {hter_vid:.4f}
APCER     : {apcer_vid:.4f}
BPCER     : {bpcer_vid:.4f}
ACER      : {acer_vid:.4f}
""")
