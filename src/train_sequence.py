"""
functions to train and evaluate a model that generates SMILES sequence from an image
Uses a CNN encoder and recurrent decoder
Model uses cross entropy loss 
"""

import argparse
import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data import get_default_transform, load_label_rows, split_labels
from src.evaluate import canonical_match, exact_match, valid_smiles_rate
from src.seq_model import ImageToSmilesModel
from src.sequence_data import SmilesSequenceDataset, get_max_sequence_length
from src.smiles_tokenizer import SmilesTokenizer


def train_sequence_model(
    dataset_name="pubchem",
    batch_size=16,
    epochs=10,
    learning_rate=0.001,
    hidden_dim=256,
    embedding_dim=128,
    decoder_type="gru",
):
    """
    function too train sequence model and evaluate on test set 
    """

    dataset_dir = os.path.join("data", dataset_name)
    image_dir = os.path.join(dataset_dir, "images")
    full_label_path = os.path.join(dataset_dir, "labels.csv")
    train_label_path = os.path.join(dataset_dir, "train_labels.csv")
    test_label_path = os.path.join(dataset_dir, "test_labels.csv")

    if not os.path.exists(train_label_path) or not os.path.exists(test_label_path):
        split_labels(full_label_path, train_label_path, test_label_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") # picks a GPU/processor
    print("Using device:", device)
    print("Dataset:", dataset_name)

    # Build a character/token vocabulary from all available labels.
    all_smiles = [row["smiles"] for row in load_label_rows(full_label_path)] 
    tokenizer = SmilesTokenizer.from_smiles_list(all_smiles)
    max_length = get_max_sequence_length(full_label_path, tokenizer)

    print("Vocabulary size:", len(tokenizer))
    print("Max sequence length:", max_length)
    print("Hidden dim:", hidden_dim)
    print("Embedding dim:", embedding_dim)
    print("Decoder type:", decoder_type)
    print("Learning rate:", learning_rate)

    transform = get_default_transform()
    train_dataset = SmilesSequenceDataset(
        image_dir=image_dir,
        label_path=train_label_path,
        tokenizer=tokenizer,
        max_length=max_length,
        transform=transform,
    )
    test_dataset = SmilesSequenceDataset(
        image_dir=image_dir,
        label_path=test_label_path,
        tokenizer=tokenizer,
        max_length=max_length,
        transform=transform,
    )

    # Dataloader handles batching/shuffling
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # convolution happens here
    model = ImageToSmilesModel(
        vocab_size=len(tokenizer),
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        decoder_type=decoder_type,
    ).to(device)

    # Ignore sequences with <PAD> 
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    final_train_token_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0 # tracks loss per run
        correct_tokens = 0
        total_tokens = 0

        for images, decoder_input, target in train_loader:
            images = images.to(device)
            decoder_input = decoder_input.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            logits = model(images, decoder_input)

            # CrossEntropyLoss expects 
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = logits.argmax(dim=-1)

            # Token accuracy is measured only on non-padding target positions.
            mask = target != tokenizer.pad_idx
            correct_tokens += ((predicted == target) & mask).sum().item()
            total_tokens += mask.sum().item()

        final_train_token_acc = (
            correct_tokens / total_tokens if total_tokens > 0 else 0.0
        )
        avg_loss = running_loss / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {avg_loss:.4f} "
            f"Train Token Accuracy: {final_train_token_acc:.4f}"
        )

    # checks test set vs training set
    metrics = evaluate_sequence_model(
        model=model,
        test_loader=test_loader,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
    )
    metrics["model"] = f"CNN-{decoder_type.upper()} Sequence"
    metrics["dataset"] = dataset_name
    metrics["decoder_type"] = decoder_type
    metrics["train_token_accuracy"] = final_train_token_acc
    metrics["vocab_size"] = len(tokenizer)
    metrics["max_length"] = max_length
    metrics["epochs"] = epochs
    metrics["batch_size"] = batch_size
    metrics["learning_rate"] = learning_rate
    metrics["hidden_dim"] = hidden_dim
    metrics["embedding_dim"] = embedding_dim

    save_metrics(metrics, "results/sequence_metrics.csv")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "token_to_idx": tokenizer.token_to_idx,
            "max_length": max_length,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim,
            "decoder_type": decoder_type,
            "dataset_name": dataset_name,
        },
        "sequence_baseline.pth",
    )
    print("Saved model to sequence_baseline.pth")


def evaluate_sequence_model(model, test_loader, tokenizer, device, max_length):
    model.eval()

    predictions = []
    targets = []
    correct_tokens = 0
    total_tokens = 0

    with torch.no_grad():
        for images, decoder_input, target in test_loader:
            images = images.to(device) # moves image tensor to processor
            decoder_input = decoder_input.to(device) # moves decoder input tensor to processor
            target = target.to(device) # moves target tensor to processor

            logits = model(images, decoder_input) # gets predictions
            predicted_tokens = logits.argmax(dim=-1) # gets predicted token IDs
            mask = target != tokenizer.pad_idx # ignores <PAD>
            correct_tokens += ((predicted_tokens == target) & mask).sum().item() # counts correct where not <PAD>
            total_tokens += mask.sum().item() # counts non <PAD> 

            # Greedy decoding simulates inference: predict one token at a time.
            generated = greedy_decode(model, images, tokenizer, device, max_length) # 

            for generated_ids, target_ids in zip(generated, target):
                predictions.append(tokenizer.decode(generated_ids))
                targets.append(tokenizer.decode(target_ids))

    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    exact_acc, _ = exact_match(predictions, targets)
    canonical_acc, _ = canonical_match(predictions, targets)
    valid_rate, _ = valid_smiles_rate(predictions)

    print("\nTest Token Accuracy:", token_accuracy)
    print("Exact Match:", exact_acc)
    print("Canonical Match:", canonical_acc)
    print("Valid SMILES Rate:", valid_rate)

    print("\nSample Predictions:")
    for i in range(min(5, len(predictions))):
        print("Predicted:", predictions[i])
        print("Target:   ", targets[i])
        print()

    return {
        "test_token_accuracy": token_accuracy,
        "exact_match": exact_acc,
        "canonical_match": canonical_acc,
        "valid_smiles_rate": valid_rate,
        "total": len(targets),
    }


def greedy_decode(model, images, tokenizer, device, max_length):
    """
    picks hte model with the highest score at each step until END
    """

    batch_size = images.size(0)
    # Start every generated sequence with the START token.
    generated = torch.full(
        (batch_size, 1),
        fill_value=tokenizer.start_idx,
        dtype=torch.long,
        device=device,
    )

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_length - 1):
        logits = model(images, generated)
        # Feed the highest-scoring token back into the decoder.
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        finished = finished | (next_token.squeeze(1) == tokenizer.end_idx)

        if finished.all():
            break

    return generated[:, 1:].cpu().tolist()


def save_metrics(metrics, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "model",
        "dataset",
        "decoder_type",
        "epochs",
        "batch_size",
        "learning_rate",
        "hidden_dim",
        "embedding_dim",
        "train_token_accuracy",
        "test_token_accuracy",
        "exact_match",
        "canonical_match",
        "valid_smiles_rate",
        "total",
        "vocab_size",
        "max_length",
    ]

    # Append each run to one shared CSV so hyperparameter experiments are easy
    # to compare in a single table.
    file_has_rows = os.path.exists(output_path) and os.path.getsize(output_path) > 0

    with open(output_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Only write the header when the file is first created.
        if not file_has_rows:
            writer.writeheader()

        writer.writerow({field: metrics.get(field, "") for field in fieldnames})

    print("Appended metrics to", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pubchem")
    parser.add_argument("--batch-size", type=int, default=16) # how many samples to process before updating wegihts
    parser.add_argument("--epochs", type=int, default=10) # how many times to iterate over entire training dataset
    parser.add_argument("--learning-rate", type=float, default=0.001) # step size for updating model weights
    parser.add_argument("--hidden-dim", type=int, default=256) # size of hidden layer in GRU decoder
    parser.add_argument("--embedding-dim", type=int, default=128) # size of token embedding vectors 
    parser.add_argument(
        "--decoder-type",
        choices=["rnn", "gru", "lstm"],
        default="gru",
    )
    args = parser.parse_args()

    train_sequence_model(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        decoder_type=args.decoder_type,
    )
