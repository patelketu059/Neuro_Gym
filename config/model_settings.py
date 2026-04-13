from pathlib import Path
import os
import torch

EMBEDDING_MODEL_ID = "nvidia/llama-nemotron-embed-vl-1b-v2"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'