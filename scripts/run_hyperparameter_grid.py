"""
Run a small hyperparameter grid for the CNN recurrent sequence models.
"""

import argparse
import itertools

from src.train_sequence import train_sequence_model


def parse_float_list(value):
    # Convert a comma-separated CLI value like "0.001,0.0005" into floats.
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value):
    # Convert a comma-separated CLI value like "128,256,512" into integers.
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value):
    # Convert a comma-separated CLI value like "rnn,gru,lstm" into strings.
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def run_grid(
    dataset,
    learning_rates,
    hidden_dims,
    embedding_dims,
    decoder_types,
    epochs,
    batch_size,
):
    """Train one model for every hyperparameter combination."""
    # Creates every learning-rate/hidden/embedding/decoder combination.
    runs = list(itertools.product(
        learning_rates,
        hidden_dims,
        embedding_dims,
        decoder_types,
    ))

    print("Total runs:", len(runs))

    for run_idx, (learning_rate, hidden_dim, embedding_dim, decoder_type) in enumerate(
        runs,
        start=1,
    ):
        print("\n" + "=" * 72)
        print(f"Run {run_idx}/{len(runs)}")
        print("Learning rate:", learning_rate)
        print("Hidden dim:", hidden_dim)
        print("Embedding dim:", embedding_dim)
        print("Decoder type:", decoder_type)
        print("=" * 72)

        # Reuse the normal training pipeline so grid runs are comparable.
        train_sequence_model(
            dataset_name=dataset,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            decoder_type=decoder_type,
        )


if __name__ == "__main__":
    # Defaults satisfy the course requirement of testing multiple parameter
    # values, while still allowing smaller runs from the command line.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pubchem")
    parser.add_argument("--learning-rates", default="0.001,0.0005,0.0001")
    parser.add_argument("--hidden-dims", default="128,256,512")
    parser.add_argument("--embedding-dims", default="128")
    parser.add_argument("--decoder-types", default="gru")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    run_grid(
        dataset=args.dataset,
        learning_rates=parse_float_list(args.learning_rates),
        hidden_dims=parse_int_list(args.hidden_dims),
        embedding_dims=parse_int_list(args.embedding_dims),
        decoder_types=parse_str_list(args.decoder_types),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
