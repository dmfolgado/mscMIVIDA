from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from pymdma.general.measures.external.piq import FrechetDistance, MultiScaleIntrinsicDistance
from pymdma.general.measures.prdc import Coverage, Density
from pymdma.image.measures.synthesis_val import GIQA, ImprovedPrecision, ImprovedRecall
from pymdma.image.models.features import ExtractorFactory
from selection_experiment import SelectionExperiment
from torch import long, tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

RESULT_FOLDER = Path("/home/duarte.folgado/TeslaVM/DeepCore/result")
DATA_PATH = Path("/home/duarte.folgado/TeslaVM/datasets/")
FEATURES_PATH = Path("/home/duarte.folgado/dev/ACHILLES/mscmivida/data/processed")
FEATURES_PATH.mkdir(parents=True, exist_ok=True)

METHOD_LIST = ["Entropy", "Uniform", "ContextualDiversity"]


def load_cifar10(data_path):
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
    """Calculates audit metrics."""
    metrics = {
        "Precision": ImprovedPrecision(k=3),
        "Recall": ImprovedRecall(k=3),
        "GIQA": GIQA(),
        "FID": FrechetDistance(),
        "MSID": MultiScaleIntrinsicDistance(),
        "Coverage": Coverage(),
        "Density": Density(),
    }
    results = {name: metric.compute(full_features, subset_features).value[0] for name, metric in metrics.items()}
    return results


def extract_and_save_features(dataset, extractor, device, filename):
    """Extracts features from a dataset with a dino model, saves and
    returns."""
    file_path = FEATURES_PATH / filename
    if file_path.exists():
        logger.info(f"Features already extracted. Loading {file_path}...")
        data = np.load(file_path)
        features, labels = data["features"], data["labels"]
        logger.info(f"Loaded features shape: {features.shape}")
        return features, labels

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    full_features, full_labels = [], []
    logger.info(f"Extracting features to {filename}...")

    for i, (img, label) in enumerate(dataloader):
        img = img.to(device)
        features = extractor(img)
        full_features.append(features.detach().cpu().numpy())
        full_labels.append(label.numpy())
        if i % 10 == 0:
            logger.info(f"Processed {i * 32 + len(img)}/{len(dataset)} images...")

    full_features = np.concatenate(full_features, axis=0)
    full_labels = np.concatenate(full_labels, axis=0)
    np.savez(FEATURES_PATH / filename, features=full_features, labels=full_labels)
    logger.info(f"{filename} feature shape: {full_features.shape}")

    return full_features, full_labels


def main():
    class_names, dst_train, dst_test = load_cifar10(DATA_PATH)
    extractor = ExtractorFactory.model_from_name(name="dino_vits8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = extractor.to(device)

    full_features, full_labels = extract_and_save_features(dst_train, extractor, device, "full_dataset.npz")

    experiment = SelectionExperiment(experiment_name="deepcore_methods_experiment")

    for method in METHOD_LIST:
        ckpt_folder = RESULT_FOLDER / method
        selection_method = experiment.add_method(method)
        fractions, accuracies = [], []
        audit_metrics = {metric: [] for metric in ["Precision", "Recall", "GIQA", "FID", "MSID", "Coverage", "Density"]}

        for f in ckpt_folder.rglob("*.ckpt"):
            fraction, accuracy = parse_experiment_checkpoint(str(f))
            logger.info(f"Processing {method} - Fraction: {fraction}%")
            fractions.append(fraction)
            accuracies.append(accuracy)

            checkpoint = torch.load(f)
            subset_indices = checkpoint["subset"]["indices"]
            subset_features = full_features[subset_indices]
            audit_results = compute_audit_metrics(full_features, subset_features)
            for key in audit_metrics:
                audit_metrics[key].append(audit_results[key])

        selection_method.set_metric("Fraction", fractions)
        selection_method.set_metric("Accuracy", accuracies)
        for metric_name, values in audit_metrics.items():
            selection_method.set_metric(metric_name, values)

    experiment.save()
    experiment.plot_all_metrics(figsize=(15, 7), save_path="all_methods_and_metrics_comparison.png")


if __name__ == "__main__":
    main()
