"""
Main training loop for BiLSTM + Attention model.

This script:
- Loads and preprocesses the data
- Builds vocabulary
- Creates PyTorch Datasets and DataLoaders
- Trains the model for N epochs
- Evaluates on validation and test sets
- Saves metrics, predictions and best model checkpoint
"""

import os
import csv
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.data.preprocess import load_and_split_data
from src.data.dataset import (
    NewsDataset,
    build_vocab,
    simple_tokenizer,
    collate_fn,
    PAD_TOKEN,
)
from src.models.bilstm_attention import BiLSTMAttentionModel
from src.training.metrics import compute_metrics
from src.utils.seed import set_seed


def train_model(config, paths, device="cuda"):
    """
    Main function that coordinates the training + evaluation.

    config: dict loaded from YAML
    paths: ProjectPaths object
    device: "cuda" or "cpu"
    """
    # Fix seed for reproducibility
    set_seed(config["seed"])

    # ------------------------------------------------------------------
    # 1. Load and split data
    # ------------------------------------------------------------------
    print("Loading and splitting data ...")
    train_df, val_df, test_df = load_and_split_data(
        data_raw_dir=paths.data_raw,
        true_file=config["data"]["true_file"],
        fake_file=config["data"]["fake_file"],
        val_size=config["data"]["val_size"],
        test_size=config["data"]["test_size"],
        random_state=config["seed"],
    )

    # ------------------------------------------------------------------
    # 2. Build vocabulary on training data only
    # ------------------------------------------------------------------
    print("🔤 Building vocabulary from training texts ...")
    vocab = build_vocab(
        train_df["text"].tolist(),
        tokenizer=simple_tokenizer,
        max_size=config["data"]["max_vocab_size"],
        min_freq=config["data"]["min_freq"],
    )
    print(f"Vocab size (including PAD/UNK): {len(vocab)}")

    pad_idx = vocab[PAD_TOKEN]
    vocab_size = len(vocab)

    # ------------------------------------------------------------------
    # 3. Create datasets and dataloaders
    # ------------------------------------------------------------------
    max_len = config["data"]["max_seq_len"]

    train_dataset = NewsDataset(
        train_df, vocab, tokenizer=simple_tokenizer, max_len=max_len
    )
    val_dataset = NewsDataset(
        val_df, vocab, tokenizer=simple_tokenizer, max_len=max_len
    )
    test_dataset = NewsDataset(
        test_df, vocab, tokenizer=simple_tokenizer, max_len=max_len
    )

    batch_size = config["training"]["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ------------------------------------------------------------------
    # 4. Set up model, optimizer, loss
    # ------------------------------------------------------------------
    print("Initialising BiLSTM + Attention model ...")
    model = BiLSTMAttentionModel(
        vocab_size=vocab_size,
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        bidirectional=config["model"]["bidirectional"],
        dropout=config["model"]["dropout"],
        pad_idx=pad_idx,
    ).to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    num_epochs = config["training"]["num_epochs"]
    best_val_f1 = 0.0
    best_model_path = os.path.join(paths.checkpoints_dir, "best_model.pt")

    # Make sure results directory exists
    os.makedirs(paths.results_dir, exist_ok=True)
    metrics_file = os.path.join(paths.results_dir, "metrics.csv")

    # Create a CSV file and write header
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "split", "accuracy", "precision", "recall", "f1"]
        )

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")

        # ------------------ Train phase ------------------
        model.train()
        train_losses = []
        all_train_labels = []
        all_train_preds = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            input_ids, lengths, labels = batch
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()

            probs, _ = model(input_ids, lengths)
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            preds = (probs.detach().cpu().numpy() >= 0.5).astype(int)
            all_train_labels.extend(labels.cpu().numpy().astype(int).tolist())
            all_train_preds.extend(preds.tolist())

        train_metrics = compute_metrics(all_train_labels, all_train_preds)

        # ------------------ Validation phase ------------------
        model.eval()
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                input_ids, lengths, labels = batch
                input_ids = input_ids.to(device)
                lengths = lengths.to(device)
                labels = labels.float().to(device)

                probs, _ = model(input_ids, lengths)
                preds = (probs.cpu().numpy() >= 0.5).astype(int)

                all_val_labels.extend(labels.cpu().numpy().astype(int).tolist())
                all_val_preds.extend(preds.tolist())

        val_metrics = compute_metrics(all_val_labels, all_val_preds)

        # Print a small summary for this epoch
        print(
            f"Train F1: {train_metrics['f1']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        # Log both train and val metrics in CSV
        with open(metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    "train",
                    train_metrics["accuracy"],
                    train_metrics["precision"],
                    train_metrics["recall"],
                    train_metrics["f1"],
                ]
            )
            writer.writerow(
                [
                    epoch,
                    "val",
                    val_metrics["accuracy"],
                    val_metrics["precision"],
                    val_metrics["recall"],
                    val_metrics["f1"],
                ]
            )

        # If this is the best validation F1 so far, save the model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_model_path)
            print(
                f"✅ New best model saved (Val F1 = {best_val_f1:.4f}) "
                f"at {best_model_path}"
            )

    # ------------------------------------------------------------------
    # 6. Load best model and evaluate on test set
    # ------------------------------------------------------------------
    print("\n🔍 Evaluating best model on test set ...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_test_labels = []
    all_test_preds = []

    os.makedirs(paths.predictions_dir, exist_ok=True)
    predictions_file = os.path.join(paths.predictions_dir, "test_predictions.csv")

    with open(predictions_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_label", "pred_label"])

        with torch.no_grad():
            idx_global = 0
            for batch in tqdm(test_loader, desc="Test"):
                input_ids, lengths, labels = batch
                input_ids = input_ids.to(device)
                lengths = lengths.to(device)
                labels = labels.float().to(device)

                probs, _ = model(input_ids, lengths)
                preds = (probs.cpu().numpy() >= 0.5).astype(int)

                for tl, pl in zip(
                    labels.cpu().numpy().astype(int), preds
                ):
                    all_test_labels.append(int(tl))
                    all_test_preds.append(int(pl))
                    writer.writerow([idx_global, int(tl), int(pl)])
                    idx_global += 1

    test_metrics = compute_metrics(all_test_labels, all_test_preds)
    print("Test metrics:", test_metrics)

    # Save test metrics in a TXT file also
    test_metrics_file = os.path.join(paths.results_dir, "test_metrics.txt")
    with open(test_metrics_file, "w") as f:
        for k, v in test_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    return test_metrics