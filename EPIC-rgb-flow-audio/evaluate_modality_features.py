"""
Evaluate modality-specific backbone embeddings with a lightweight classifier.

This script mirrors the feature extraction logic used during training and allows you to
select one or more modalities (video / flow / audio) and evaluate their embeddings
individually (common half, specific half, or the full vector). It loads the pretrained
checkpoint, computes frozen backbone embeddings for the requested train/test domains,
trains a small classifier head on the train split, and reports the classification
accuracy on the test split. When multiple modalities are selected, the script also
trains a classifier on the concatenated features across modalities.

Example
-------
python evaluate_modality_features.py \
    --modality video \
    --embedding-types common specific \
    --checkpoint-path models/your_experiment/checkpoint.pt \
    --train-domain D1 D2 \
    --test-domain D3 \
    --datapath /path/to/EPIC-KITCHENS/
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from argparse import Namespace

from extract_features import load_models
from dataloader_EPIC_MERDG_lmdb_optimized import (
    EPICDOMAIN_OPTIMIZED_LMDB,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate modality-specific embeddings with a new classifier head",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Modality & embedding selection
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        required=True,
        choices=["video", "flow", "audio"],
        help="One or more modalities to evaluate.",
    )
    parser.add_argument(
        "--embedding-types",
        type=str,
        nargs="+",
        default=["common", "specific"],
        choices=["common", "specific", "full", "all"],
        help="Which embedding splits to evaluate. Use 'all' to evaluate every split.",
    )

    # Data/domain configuration
    parser.add_argument(
        "--train-domain",
        "-s",
        type=str,
        nargs="+",
        required=True,
        help="Source domain(s) for training the classifier.",
    )
    parser.add_argument(
        "--test-domain",
        "-t",
        type=str,
        nargs="+",
        required=True,
        help="Target domain(s) for evaluation.",
    )
    parser.add_argument(
        "--split-train",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split used for training features.",
    )
    parser.add_argument(
        "--split-test",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split used for evaluation features.",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="/path/to/EPIC-KITCHENS/",
        help="Path to EPIC-KITCHENS dataset root.",
    )
    parser.add_argument(
        "--use-lmdb",
        action="store_true",
        help="Use the optimized LMDB dataloader (requires rgb.lmdb/flow.lmdb).",
    )
    parser.add_argument(
        "--lmdb-path",
        type=str,
        default="/home/yavuz/data/EPIC/frames_rgb_flow_lmdb/",
        help="Path to LMDB directory when --use-lmdb is enabled.",
    )

    # Checkpoint & classifier training
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Checkpoint file with trained model weights.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of epochs for classifier training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for feature extraction.",
    )
    parser.add_argument(
        "--classifier-batch-size",
        type=int,
        default=256,
        help="Batch size for classifier training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the classifier head.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for the classifier optimizer.",
    )
    parser.add_argument(
        "--classifier-hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension of the classifier head (set 0 for linear).",
    )
    parser.add_argument(
        "--classifier-dropout",
        type=float,
        default=0.3,
        help="Dropout applied after the hidden layer (ignored when hidden dim=0).",
    )

    # Misc
    parser.add_argument(
        "--num-workers",
        type=int,
        default=12,
        help="Number of workers for feature extraction dataloaders.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Optional path to save metrics as JSON.",
    )
    parser.add_argument(
        "--print-progress",
        action="store_true",
        help="Show tqdm progress bars during feature extraction.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloader(
    split: str,
    domains: List[str],
    cfg,
    cfg_flow,
    args: argparse.Namespace,
) -> DataLoader:
    dataset = EPICDOMAIN_OPTIMIZED_LMDB(
        split=split,
        domain=domains,
        cfg=cfg,
        cfg_flow=cfg_flow,
    datapath=args.datapath,
    use_video=("video" in args.modalities),
    use_flow=("flow" in args.modalities),
    use_audio=("audio" in args.modalities),
        use_lmdb=args.use_lmdb,
        lmdb_path=args.lmdb_path,
    )

    shuffle = split == args.split_train
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )
    return loader


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor)
    return tensor.to(device, non_blocking=True)


def collect_embeddings(
    dataloader: DataLoader,
    modalities: List[str],
    device: torch.device,
    model_bundle: Tuple,
    show_progress: bool,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], torch.Tensor]:
    model, model_flow, audio_model, audio_cls_model = model_bundle

    iterator = dataloader
    if show_progress:
        iterator = tqdm(dataloader, desc="Extracting embeddings")

    features_per_mod: Dict[str, List[torch.Tensor]] = {mod: [] for mod in modalities}
    labels: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in iterator:
            clip, flow, spectrogram, label_batch, _ = batch
            label_batch = label_batch.long()

            if not labels:
                labels.append(label_batch.detach().cpu())
            else:
                labels.append(label_batch.detach().cpu())

            if "video" in modalities:
                clip_tensor = clip["imgs"].squeeze(1).to(device, non_blocking=True)
                x_slow, x_fast = model.module.backbone.get_feature(clip_tensor)
                v_feat = model.module.backbone.get_predict((x_slow, x_fast))
                _, v_embedding = model.module.cls_head(v_feat)
                features_per_mod["video"].append(v_embedding.detach().cpu())

            if "flow" in modalities:
                flow_tensor = flow["imgs"].squeeze(1).to(device, non_blocking=True)
                f_feat = model_flow.module.backbone.get_feature(flow_tensor)
                f_feat = model_flow.module.backbone.get_predict(f_feat)
                _, f_embedding = model_flow.module.cls_head(f_feat)
                features_per_mod["flow"].append(f_embedding.detach().cpu())

            if "audio" in modalities:
                spectrogram_tensor = spectrogram.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)
                _, audio_feat, _ = audio_model(spectrogram_tensor)
                _, a_embedding = audio_cls_model(audio_feat)
                features_per_mod["audio"].append(a_embedding.detach().cpu())

    if not labels:
        raise RuntimeError("No features were collected. Check dataloader/modality configuration.")

    labels_tensor = torch.cat(labels, dim=0)

    embeddings: Dict[str, Dict[str, torch.Tensor]] = {}
    for modality, feat_list in features_per_mod.items():
        if not feat_list:
            raise RuntimeError(f"No features collected for modality '{modality}'.")
        feat_tensor = torch.cat(feat_list, dim=0)
        if feat_tensor.shape[1] % 2 != 0:
            raise ValueError(
                f"Expected even feature dimensionality for common/specific split, got {feat_tensor.shape[1]}"
            )
        half_dim = feat_tensor.shape[1] // 2
        embeddings[modality] = {
            "full": feat_tensor,
            "common": feat_tensor[:, :half_dim],
            "specific": feat_tensor[:, half_dim:],
        }

    return embeddings, labels_tensor


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        if hidden_dim > 0:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_dim, num_classes),
                ]
            )
        else:
            layers.append(nn.Linear(input_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate_accuracy(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> float:
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            logits = model(batch_x)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total if total > 0 else 0.0


def train_classifier(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    num_classes = int(train_labels.max().item() + 1)
    input_dim = train_features.shape[1]

    classifier = LinearClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=args.classifier_hidden_dim,
        dropout=args.classifier_dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.classifier_batch_size,
        shuffle=True,
        drop_last=False,
    )

    best_test_acc = 0.0
    best_state = None
    history = []

    for epoch in range(args.epochs):
        classifier.train()
        epoch_loss = 0.0
        total = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = classifier(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_y.size(0)
            total += batch_y.size(0)

        avg_loss = epoch_loss / max(total, 1)
        train_acc = evaluate_accuracy(
            classifier, train_features, train_labels, device, args.classifier_batch_size
        )
        test_acc = evaluate_accuracy(
            classifier, test_features, test_labels, device, args.classifier_batch_size
        )

        history.append({"epoch": epoch, "train_loss": avg_loss, "train_acc": train_acc, "test_acc": test_acc})

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {k: v.cpu() for k, v in classifier.state_dict().items()}

    if best_state is not None:
        classifier.load_state_dict(best_state)

    final_train_acc = evaluate_accuracy(
        classifier, train_features, train_labels, device, args.classifier_batch_size
    )
    final_test_acc = evaluate_accuracy(
        classifier, test_features, test_labels, device, args.classifier_batch_size
    )

    return {
        "train_accuracy": final_train_acc,
        "test_accuracy": final_test_acc,
        "best_test_accuracy": best_test_acc,
        "history": history,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if "all" in args.embedding_types:
        args.embedding_types = ["common", "specific", "full"]

    device = torch.device(args.device)

    model_args = Namespace(
        use_video="video" in args.modalities,
        use_flow="flow" in args.modalities,
        use_audio="audio" in args.modalities,
        checkpoint_path=args.checkpoint_path,
    )

    model, model_flow, audio_model, audio_cls_model, cfg, cfg_flow = load_models(model_args, device)

    if model is not None:
        model.eval()
    if model_flow is not None:
        model_flow.eval()
    if audio_model is not None:
        audio_model.eval()
    if audio_cls_model is not None:
        audio_cls_model.eval()

    train_loader = build_dataloader(args.split_train, args.train_domain, cfg, cfg_flow, args)
    test_loader = build_dataloader(args.split_test, args.test_domain, cfg, cfg_flow, args)

    train_embeddings, train_labels = collect_embeddings(
        train_loader,
        args.modalities,
        device,
        (model, model_flow, audio_model, audio_cls_model),
        args.print_progress,
    )
    test_embeddings, test_labels = collect_embeddings(
        test_loader,
        args.modalities,
        device,
        (model, model_flow, audio_model, audio_cls_model),
        args.print_progress,
    )

    # Convert to float/long tensors once to avoid repeated casts
    for modality in train_embeddings:
        for key in train_embeddings[modality]:
            train_embeddings[modality][key] = train_embeddings[modality][key].float()
    train_labels = train_labels.long()
    for modality in test_embeddings:
        for key in test_embeddings[modality]:
            test_embeddings[modality][key] = test_embeddings[modality][key].float()
    test_labels = test_labels.long()

    print("\n==================== Evaluation ====================")
    print(f"Modalities        : {args.modalities}")
    print(f"Train split/domains: {args.split_train} / {args.train_domain}")
    print(f"Test split/domains : {args.split_test} / {args.test_domain}")
    print(f"Samples (train/test): {train_labels.shape[0]} / {test_labels.shape[0]}")

    results: Dict[str, Dict[str, Dict[str, float]]] = {"modalities": {}, "combined": {}}

    # Per-modality evaluations
    for modality in args.modalities:
        print(f"\n===== Modality: {modality} =====")
        results["modalities"][modality] = {}
        for emb_type in args.embedding_types:
            train_feats = train_embeddings[modality][emb_type]
            test_feats = test_embeddings[modality][emb_type]
            print(f"Embedding '{emb_type}' -> train {tuple(train_feats.shape)}, test {tuple(test_feats.shape)}")

            metrics = train_classifier(train_feats, train_labels, test_feats, test_labels, args, device)
            results["modalities"][modality][emb_type] = {
                "train_accuracy": metrics["train_accuracy"],
                "test_accuracy": metrics["test_accuracy"],
                "best_test_accuracy": metrics["best_test_accuracy"],
            }

            print(
                f"  Train acc: {metrics['train_accuracy']*100:6.2f}% | "
                f"Test acc: {metrics['test_accuracy']*100:6.2f}% | "
                f"Best test acc: {metrics['best_test_accuracy']*100:6.2f}%"
            )

    # Combined multi-modal evaluation (concatenate embeddings across modalities)
    if len(args.modalities) > 1:
        print("\n===== Combined Modalities =====")
        for emb_type in args.embedding_types:
            combined_train = torch.cat(
                [train_embeddings[mod][emb_type] for mod in args.modalities], dim=1
            )
            combined_test = torch.cat(
                [test_embeddings[mod][emb_type] for mod in args.modalities], dim=1
            )
            print(
                f"Embedding '{emb_type}' -> train {tuple(combined_train.shape)}, "
                f"test {tuple(combined_test.shape)}"
            )

            metrics = train_classifier(combined_train, train_labels, combined_test, test_labels, args, device)
            results.setdefault("combined", {})[emb_type] = {
                "train_accuracy": metrics["train_accuracy"],
                "test_accuracy": metrics["test_accuracy"],
                "best_test_accuracy": metrics["best_test_accuracy"],
            }

            print(
                f"  Train acc: {metrics['train_accuracy']*100:6.2f}% | "
                f"Test acc: {metrics['test_accuracy']*100:6.2f}% | "
                f"Best test acc: {metrics['best_test_accuracy']*100:6.2f}%"
            )

    if args.save_results is not None:
        os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
        with open(args.save_results, "w") as f:
            json.dump(
                {
                    "modalities": args.modalities,
                    "embedding_types": args.embedding_types,
                    "train_domain": args.train_domain,
                    "test_domain": args.test_domain,
                    "train_split": args.split_train,
                    "test_split": args.split_test,
                    "checkpoint": args.checkpoint_path,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\nMetrics saved to {args.save_results}")


if __name__ == "__main__":
    main()
