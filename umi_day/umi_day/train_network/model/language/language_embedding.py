import torch
from torch import nn
from typing import List
from sentence_transformers import SentenceTransformer
from typing import List

class LanguageEmbedding(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.model = SentenceTransformer(model_name, device='cpu')

        # freeze the language embedding
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    @torch.inference_mode() 
    def embed(self, language_batch: List[str]):
        embeddings = self.model.encode(language_batch, show_progress_bar=False)
        return embeddings
    