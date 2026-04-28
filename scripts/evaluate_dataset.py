"""
Evaluate a saved CNN-RNN/CNN-GRU/CNN-LSTM checkpoint on an external dataset.

Use this for datasets like EPA or DECIMER when the model was trained on
PubChem and the new dataset is only being used for testing.
"""

import argparse
import csv
import os
import tempfile

import torch
from torch.utils.data import DataLoader

from src.data import get_default_transform, load_label_rows
from src.evaluate import canonical_match, exact_match, valid_smiles_rate
from src.seq_model import ImageToSmilesModel
from src.sequence_data import SmilesSequenceDataset
from src.smiles_tokenizer import SmilesTokenizer
from src.train_sequence import greedy_decode


def get_default_checkpoint_path(decoder_type):
    """Build the default checkpoint path for a decoder type."""
    return f"sequence_{decoder_type}.pth"


def get_device():
    """Pick the best available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_external_metrics(metrics, output_path):
    """Append one external-evaluation row to a shared CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "model",
        "checkpoint",
        "train_dataset",
        "eval_dataset",
        "decoder_type",
        "batch_size",
        "limit",
        "test_token_accuracy",
        "exact_match",
        "canonical_match",
        "valid_smiles_rate",
        "total",
        "vocab_size",
        "max_length",
        "hidden_dim",
        "embedding_dim",
    ]

    file_has_rows = os.path.exists(output_path) and os.path.getsize(output_path) > 0

    with open(output_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_has_rows:
            writer.writeheader()

        writer.writerow({field: metrics.get(field, "") for field in fieldnames})

    print("Appended external metrics to", output_path)


def evaluate_dataset(
    dataset_name="epa",
    checkpoint_path=None,
    decoder_type="gru",
    batch_size=16,
    limit=None,
    output_path="results/external_metrics.csv",
):
    """Load a trained checkpoint and evaluate it on a dataset."""
    if checkpoint_path is None:
        checkpoint_path = get_default_checkpoint_path(decoder_type)

    dataset_dir = os.path.join("data", dataset_name)
    image_dir = os.path.join(dataset_dir, "images")
    label_path = os.path.join(dataset_dir, "labels.csv")

    device = get_device()
    print("Using device:", device)
    print("Evaluation dataset:", dataset_name)
    print("Checkpoint:", checkpoint_path)
    print("Limit:", limit if limit is not None else "all")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = SmilesTokenizer(checkpoint["token_to_idx"])
    max_length = checkpoint["max_length"]
    hidden_dim = checkpoint["hidden_dim"]
    embedding_dim = checkpoint["embedding_dim"]
    decoder_type = checkpoint.get("decoder_type", "gru")

    model = ImageToSmilesModel(
        vocab_size=len(tokenizer),
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        decoder_type=decoder_type,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = load_label_rows(label_path)
    if limit is not None:
        rows = rows[:limit]

    eval_label_path = label_path
    temp_label_file = None

    if limit is not None:
        temp_label_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".csv",
            delete=False,
        )
        writer = csv.DictWriter(temp_label_file, fieldnames=["image", "smiles"])
        writer.writeheader()
        writer.writerows(rows)
        temp_label_file.close()
        eval_label_path = temp_label_file.name

    dataset = SmilesSequenceDataset(
        image_dir=image_dir,
        label_path=eval_label_path,
        tokenizer=tokenizer,
        max_length=max_length,
        transform=get_default_transform(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    targets = []
    correct_tokens = 0
    total_tokens = 0
    row_index = 0

    try:
        with torch.no_grad():
            for images, decoder_input, target in loader:
                images = images.to(device)
                decoder_input = decoder_input.to(device)
                target = target.to(device)

                # Teacher-forced token accuracy shows whether the model scores
                # the correct next tokens when given correct previous tokens.
                logits = model(images, decoder_input)
                predicted_tokens = logits.argmax(dim=-1)
                mask = target != tokenizer.pad_idx
                correct_tokens += ((predicted_tokens == target) & mask).sum().item()
                total_tokens += mask.sum().item()

                # Greedy decoding simulates real inference on external images.
                generated = greedy_decode(model, images, tokenizer, device, max_length)
                batch_predictions = [tokenizer.decode(ids) for ids in generated]
                batch_targets = [
                    row["smiles"]
                    for row in rows[row_index: row_index + len(batch_predictions)]
                ]

                predictions.extend(batch_predictions)
                targets.extend(batch_targets)
                row_index += len(batch_predictions)
    finally:
        if temp_label_file is not None:
            os.remove(temp_label_file.name)

    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    exact_acc, _ = exact_match(predictions, targets)
    canonical_acc, _ = canonical_match(predictions, targets)
    valid_rate, _ = valid_smiles_rate(predictions)

    print("\nExternal Test Token Accuracy:", token_accuracy)
    print("Exact Match:", exact_acc)
    print("Canonical Match:", canonical_acc)
    print("Valid SMILES Rate:", valid_rate)

    print("\nSample Predictions:")
    for i in range(min(5, len(predictions))):
        print("Predicted:", predictions[i])
        print("Target:   ", targets[i])
        print()

    metrics = {
        "model": f"CNN-{decoder_type.upper()} Sequence",
        "checkpoint": checkpoint_path,
        "train_dataset": checkpoint.get("dataset_name", ""),
        "eval_dataset": dataset_name,
        "decoder_type": decoder_type,
        "batch_size": batch_size,
        "limit": limit if limit is not None else "",
        "test_token_accuracy": token_accuracy,
        "exact_match": exact_acc,
        "canonical_match": canonical_acc,
        "valid_smiles_rate": valid_rate,
        "total": len(targets),
        "vocab_size": len(tokenizer),
        "max_length": max_length,
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
    }
    save_external_metrics(metrics, output_path)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="epa")
    parser.add_argument(
        "--decoder-type",
        choices=["rnn", "gru", "lstm"],
        default="gru",
        help="Decoder checkpoint to use when --checkpoint is not provided.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint override. Defaults to sequence_<decoder-type>.pth.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="results/external_metrics.csv")
    args = parser.parse_args()

    evaluate_dataset(
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        decoder_type=args.decoder_type,
        batch_size=args.batch_size,
        limit=args.limit,
        output_path=args.output,
    )
