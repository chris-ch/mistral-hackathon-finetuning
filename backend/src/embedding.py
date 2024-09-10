"""_summary_

    Returns:
        _type_: _description_
"""
import logging
import sys
from typing import List

from chromadb import EmbeddingFunction
from chromadb.api.types import Document, Embedding

from mistralai import Mistral


class MistralEmbeddingFunction(EmbeddingFunction):
    """_summary_

    Args:
        EmbeddingFunction (_type_): _description_
    """
    def __init__(self, api_key: str, model_deployment: str):
        self.client = Mistral(api_key=api_key)
        self.model = model_deployment

    def __call__(self, docs: List[Document]) -> List[Embedding]:
        embeddings_batch_response = self.client.embeddings.create(model=self.model, inputs=docs)
        #TODO: make sure that the order is preserved?!
        return [entry.embedding for entry in embeddings_batch_response.data]


class EmbeddingModel:
    """_summary_
    """
    def __init__(self, model_deployment: str, api_key: str, batch_size: int = 1):
        """Use API calls to embed content"""
        self.embedding_fun = MistralEmbeddingFunction(
                api_key=api_key,
                model_deployment=model_deployment,
            )
        self.batch_size = batch_size

    def embed(self, docs: List[Document])-> List[Embedding]:
        """_summary_

        Args:
            doc (Documents): _description_

        Returns:
            _type_: _description_
        """
        count_batches = len(docs) // self.batch_size
        if len(docs) % self.batch_size != 0:
            count_batches += 1

        logging.info("processing %s batches", count_batches)
        embeddings = []
        for batch_idx in range(count_batches):
            idx_start = batch_idx * self.batch_size
            idx_end = (batch_idx + 1) * self.batch_size
            batch = docs[idx_start:idx_end]
            embeddings += self.embedding_fun(batch)

            # Progress indicator
            progress = (batch_idx + 1) / count_batches
            bar_length = 30
            filled_length = int(bar_length * progress)
            progress_bar = '=' * filled_length + '-' * (bar_length - filled_length)

            sys.stdout.write(f'\rProgress: [{progress_bar}] {progress:.1%} ({batch_idx + 1}/{count_batches})')
            sys.stdout.flush()

        # Print a newline after the loop completes
        print()

        return embeddings
