"""
Embed protein sequences from a FASTA file using ESM models.

This script reads protein sequences from a FASTA file, generates embeddings
using a specified ESM (Evolutionary Scale Modeling) model, and saves the
results to a Parquet file.

Command-line Arguments:
    -i, --input (str): Path to the input FASTA file containing protein sequences.
                       The file must be readable.
    -o, --output (str): Path to the output Parquet file to save the embeddings.
    -m, --model (str): Name of the ESM model to use for embedding.
                       Available options: esmc_300m, esmc_600m

Raises:
    AssertionError: If the input FASTA file is not readable or if the specified
                    model is not in the list of available models.

Output Format:
    The output Parquet file contains the following columns:
    - seq_id (str): The sequence identifier from the FASTA file
    - position (int): The position index within the sequence
    - Numbered columns (0 to N): Embedding for each position

Notes:
    - The script automatically uses CUDA GPU if available, otherwise falls back
      to CPU.
    - Each embedding is converted to a NumPy array and squeezed to remove
      singleton dimensions.
    - If no embeddings are generated, the script exits with code 1.
"""
import os
import sys
import argparse
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import torch
from Bio import SeqIO
from tqdm import tqdm
import pandas as pd
import numpy as np

AVAILABLE_MODELS = ["esmc_300m", "esmc_600m"]


parser = argparse.ArgumentParser(
    description="Embed protein sequences from a FASTA file."
)
parser.add_argument(
    "-i",
    "--input",
    type=str,
    required=True,
    help="Path to the input FASTA file containing protein sequences.",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=True,
    help="Path to the output Parquet file to save the embeddings.",
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    required=True,
    help=f"Name of the ESM model to use for embedding ({', '.join(AVAILABLE_MODELS)}).",
)
args = parser.parse_args()

assert os.path.isfile(args.input) and os.access(args.input, os.R_OK), (
    f"FASTA file '{args.input}' is not readable."
)
assert args.model in AVAILABLE_MODELS, (
    f"Model '{args.model}' is not available. Choose from: {', '.join(AVAILABLE_MODELS)}."
)

FASTA_FILE = args.input
OUT_FILE = args.output
MODEL = args.model


def embed_protein(sequence: str):
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)
    logits_output = client.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    return logits_output.embeddings


has_cuda = torch.cuda.is_available()
client = ESMC.from_pretrained(MODEL).to("cuda" if torch.cuda.is_available() else "cpu")

embeddings = {}

for record in tqdm(SeqIO.parse(FASTA_FILE, "fasta")):
    seq_id = record.id
    sequence = str(record.seq)
    embeddings[seq_id] = embed_protein(sequence)

embeddings_np = {k: v.cpu().numpy().squeeze() for k, v in embeddings.items()}

if len(embeddings_np) == 0:
    print("No embeddings were generated. Exiting.")
    sys.exit(1)

arrays = np.vstack([arr for arr in embeddings_np.values()])
keys = np.repeat(
    list(embeddings_np.keys()), [arr.shape[0] for arr in embeddings_np.values()]
)
position = np.concatenate([np.arange(arr.shape[0]) for arr in embeddings_np.values()])

df = pd.DataFrame(arrays, columns=[str(i) for i in range(arrays.shape[1])])
df.insert(0, "seq_id", keys)
df.insert(1, "position", position)
df.columns = df.columns.astype(str)

df.to_parquet(OUT_FILE, index=True)
