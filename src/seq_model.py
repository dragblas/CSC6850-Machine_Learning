"""
CNN encoder + GRU decoder baseline for image-to-SMILES generation.
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.features = nn.Sequential(
            # Convert the 224 x 224 molecule image into compact visual features.
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), # 224 x 224 x 3 -> 224 x 224 x 32
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 112 x 112 x 32 -> 112 x 112 x 64
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 56 x 56 x 64 -> 56 x 56 x 128
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # 28 x 28 x 128 -> 28 x 28 x 256
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)), 
        )
        # Match the CNN feature size to the GRU hidden state size.
        self.projection = nn.Linear(256, hidden_dim)

    def forward(self, images):
        features = self.features(images)
        features = features.flatten(start_dim=1)
        return torch.tanh(self.projection(features))


class ImageToSmilesModel(nn.Module):
    """
    CNN encoder + GRU decoder for image-to-SMILES generation.
    """

    def __init__(
        self,
        vocab_size,
        hidden_dim=256,
        embedding_dim=128,
    ):
        super().__init__()

        self.encoder = CNNEncoder(hidden_dim=hidden_dim) # class above where CNN processes the image and outputs a feature vector of size hidden_dim

        # TOKENS TO VECTORS
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.decoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, decoder_input):
        image_features = self.encoder(images)
        # Use the image embedding as the initial hidden state for the decoder.
        initial_hidden = image_features.unsqueeze(0)

        embedded_tokens = self.embedding(decoder_input)
        decoder_output, _ = self.decoder(embedded_tokens, initial_hidden)

        # Output one vocabulary-sized score vector for each sequence position.
        return self.output(decoder_output)
