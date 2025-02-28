from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from loguru import logger
from pymdma.image.measures.synthesis_val import ImprovedPrecision, ImprovedRecall
from pymdma.general.measures.external.piq import FrechetDistance, MultiScaleIntrinsicDistance
from pymdma.general.measures.prdc import Coverage, Density
from pymdma.image.measures.synthesis_val import GIQA
from pymdma.image.models.features import ExtractorFactory
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch import tensor, long


CKPT_FOLDER = Path("/home/barbara/DeepCore/result/Entropy")
DATA_PATH = Path("/home/barbara/DeepCore/datasets/")
FEATURES_PATH = Path("src/mscmivida/evaluation/28fev")
FEATURES_PATH.mkdir(parents=True, exist_ok=True)

def CIFAR10(data_path):
    channel = 3
    im_size = (32, 32)
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
    dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return class_names, dst_train, dst_test


def parse_experiment_checkpoint(path: str) -> tuple[float, float]:
    """Parses experiment checkpoint filename to extract fraction and
    accuracy."""
    try:
        parts = Path(path).stem.split("_")
        accuracy = float(parts[-1][:5])
        fraction = float(parts[-2]) * 100
        return fraction, accuracy
    except (IndexError, ValueError) as e:
        logger.error(f"Failed to parse checkpoint path: {path} | Error: {e}")
        return 0.0, 0.0

def compute_audit_metrics(full_features, subset_features):
    """Calculates audit metrics"""
    ip, ir, giqa = ImprovedPrecision(k=3), ImprovedRecall(k=3), GIQA()
    fid, msid, coverage, density = FrechetDistance(), MultiScaleIntrinsicDistance(), Coverage(), Density()
    print("Improved Precision...")
    ip_result = ip.compute(full_features, subset_features).value
    print("Improved Recall...")
    ir_result = ir.compute(full_features, subset_features).value
    print("GIQA...")
    giqa_result = giqa.compute(full_features, subset_features).value
    print("FID...")
    fid_result = fid.compute(full_features, subset_features).value
    print("MSID...")
    msid_result = msid.compute(full_features, subset_features).value
    print("Coverage...")
    coverage_result = coverage.compute(full_features, subset_features).value
    print("Density...")
    density_result = density.compute(full_features, subset_features).value
    
    return {
        "Precision": ip_result[0], "Recall": ir_result[0], "GIQA": giqa_result[0],
        "FID": fid_result, "MSID": msid_result, "Coverage": coverage_result, "Density": density_result
    }


def load_subset_features(full_features, indices):
    """Obtains the features of a subset from the indexes"""
    subset_features = full_features[indices]
    print(f"Subset shape: {subset_features.shape}")
    return subset_features

def extract_and_save_features(dataset, extractor, device, filename):
    """Extracts features from a dataset with a dino model, saves and returns"""
    file_path = FEATURES_PATH / filename
    if file_path.exists():
        print(f"Features already extracted. Loading {file_path}...")
        data = np.load(file_path)
        features, labels = data["features"], data["labels"]
        print(f"Loaded features shape: {features.shape}")
        return features, labels

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False) #o dataloader carrega imagens em batches
    full_features, full_labels = [], []
    print(f"Extracting features to {filename}...")

    for i, (img, label) in enumerate(dataloader): #percorre o dataset e extrai features
        """i é nº de batch, Cada img contém um batch de imagens - tensor de tamanho (batch size, 3RGB, H, W dimensões da imagem)"""
        img = img.to(device)
        features = extractor(img) #extração com modelo pre treinado (dino), as features extraídas são tensores
        full_features.append(features.detach().cpu().numpy()) #salva os tensores extraídos na CPU como NumPy arrays
        full_labels.append(label.numpy())

        if i % 10 == 0:
            print(f"Processed {i * 32 + len(img)}/{len(dataset)} images...")

    full_features = np.concatenate(full_features, axis=0)
    full_labels = np.concatenate(full_labels, axis=0)
    np.savez(FEATURES_PATH / filename, features=full_features, labels=full_labels)
    print(f"{filename} feature shape: {full_features.shape}")

    return full_features, full_labels



def main(): 
    class_names, dst_train, dst_test = CIFAR10(DATA_PATH)
    extractor = ExtractorFactory.model_from_name(name="dino_vits8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    extractor = extractor.to(device)

    full_features, full_labels = extract_and_save_features(dst_train, extractor, device, "full_dataset.npz")
    evaluation_metrics_db = {"Fraction": [], "Accuracy": [], "Precision": [], "Recall": [], "GIQA": [], "FID": [], "MSID": [], "Coverage": [], "Density": []}

    for f in CKPT_FOLDER.rglob("*.ckpt"):
        fraction, accuracy = parse_experiment_checkpoint(str(f))
        evaluation_metrics_db["Fraction"].append(fraction)
        evaluation_metrics_db["Accuracy"].append(accuracy)

        print(f"Processing subset for fraction: {fraction}%")
        checkpoint = torch.load(f)
        subset_indices = checkpoint["subset"]["indices"]
        subset_features = load_subset_features(full_features, subset_indices)
        print(f"Features from fraction {fraction}% loaded")
        audit_results = compute_audit_metrics(full_features, subset_features)
        for key in audit_results:
            evaluation_metrics_db[key].append(audit_results[key])


    evaluation_metrics_db = pd.DataFrame(evaluation_metrics_db).sort_values("Fraction")
    print(evaluation_metrics_db)
    evaluation_metrics_db.to_csv(FEATURES_PATH / "audit_results.csv", index=False)

    # Plot Fraction vs Accuracy
    fig, ax = plt.subplots()
    ax.plot(evaluation_metrics_db["Fraction"], evaluation_metrics_db["Accuracy"], "o-")
    ax.set_xlabel("Fraction", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    plt.show(block=False)

if __name__ == "__main__":
    main()
