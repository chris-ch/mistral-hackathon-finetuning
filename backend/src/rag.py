"""
"""
from enum import Enum
import logging
from typing import List, Dict, Optional, Tuple

import sys
import json
from time import time

import chromadb
from chromadb.api.types import QueryResult
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb import Documents, EmbeddingFunction, Embeddings

from mistralai import Mistral, SystemMessage, UserMessage

from utils import Case
from prompts import CASE_PLACEHOLDER, SUPPORTING_CONTENT_PLACEHOLDER


class LegalType(Enum):
    """_summary_

    Args:
        Enum (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    PRIVATE = "private"
    CRIMINAL = "criminal"
    PUBLIC = "state"

    @staticmethod
    def from_string(value: str):
        """_summary_

        Args:
            value (str): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        try:
            return LegalType(value)
        except ValueError as exc:
            raise ValueError(f"'{value}' is not a valid LegalType") from exc


class CompletionModel:
    """_summary_
    """
    def __init__(
            self,
            api_key: str,
            model_deployment: str
        ):
        """Class interfacing with the model deployments"""
        self.model = model_deployment
        self.client = Mistral(api_key=api_key)

    def call(
        self,
        system_user_q_and_a: Tuple[SystemMessage, UserMessage],
        temperature: Optional[float],
    ) -> str:
        """Send request to LLM
        
        Parameters:
        -----------
            messages: conversation with the LLM, can include system messages as well as the history
            temperature: impact the imagination and variability of the LLm answers

        Returns:
        --------
            answer: the content of the reponse of the LLM
        """
        response = self.client.chat.complete(
            model=self.model,
            messages=list(system_user_q_and_a),
            max_tokens=None,
            temperature=temperature,
        )

        # TODO: might contain function call?
        answer = response.choices[0].message.content
        return answer if answer is not None else "Failed to complete"


class MistralEmbeddingFunction(EmbeddingFunction):
    """_summary_

    Args:
        EmbeddingFunction (_type_): _description_
    """
    def __init__(self, api_key: str, model_deployment: str):
        self.client = Mistral(api_key=api_key)
        self.model = model_deployment

    def __call__(self, doc: Documents) -> Embeddings:
        embeddings_batch_response = self.client.embeddings.create(model=self.model, inputs=doc)
        #TODO: make sure that the order is preserved?!
        return [entry.embedding for entry in embeddings_batch_response.data]


class EmbeddingModel:
    """_summary_
    """
    def __init__(self, model_deployment: str, api_key: str):
        """Use API calls to embed content"""
        self.embedding_fun = MistralEmbeddingFunction(
                api_key=api_key,
                model_deployment=model_deployment,
            )
        self.batch_size = 1

    def embed(self, doc: Documents):
        """_summary_

        Args:
            doc (Documents): _description_

        Returns:
            _type_: _description_
        """
        nb_batches = len(doc) // self.batch_size
        if len(doc) % self.batch_size != 0:
            nb_batches += 1

        logging.info("created %s batches", nb_batches)
        embeddings = []
        for batch_idx in range(nb_batches):
            idx_start = batch_idx * self.batch_size
            idx_end = (batch_idx + 1) * self.batch_size
            batch = doc[idx_start:idx_end]
            embeddings += self.embedding_fun(batch)

            # Progress indicator
            progress = (batch_idx + 1) / nb_batches
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '=' * filled_length + '-' * (bar_length - filled_length)

            sys.stdout.write(f'\rProgress: [{bar}] {progress:.1%} ({batch_idx + 1}/{nb_batches})')
            sys.stdout.flush()

        # Print a newline after the loop completes
        print()

        return embeddings


class RAGModel:
    """_summary_
    """
    def __init__(self, legal_type: LegalType, config: Dict):
        """Model responsible for consuming the data to build a knowledge database"""
        self.config = config
        self.prompt_system = config["prompt_system"]
        self.prompt_template = config["prompt_template"]

        # Embedding model, convert natural langage to vector
        self.embedding_model = EmbeddingModel(
            model_deployment=config["embedding_model_deployment"],
            api_key=config["embedding_api_key"],
        )

        logging.info("creating Vector DB persistent client for expert '%s'", legal_type.value)

        # Vector database/Search index
        self.db_client = chromadb.PersistentClient(
            path="chroma",
            settings=Settings(anonymized_telemetry=False),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )
        #TODO: the name of the collection could actually be a parameter of predict,
        # to allow the model to switch between vector db
        self.vectordb = self.db_client.get_or_create_collection(name=legal_type.value)
        # Empty collection, need to populate it
        if self.vectordb.count() == 0:
            logging.error("The Vector DB must be populated using the 'init-vector-db' script")
            sys.exit(0)

        logging.info('creating completion model')

        # Completion model, answer request based on supporting content
        self.completion_model = CompletionModel(
            api_key=config["completion_api_key"],
            model_deployment=config["completion_model_deployment"],
        )

    def predict(self, case: Case):
        """
        Execute all the steps of the RAG logic:
            1. embed the query
            2. retrieve the supporting content
            3. update the prompt with the information
            4. return the completion model reponse
        """
        logging.info("starting predict phase")
        relevant_chunks = self._retrieve_supporting_content(case.description)

        # convert relevant chunks to a list of string
        relevant_chunks_content = relevant_chunks["documents"]
        if relevant_chunks_content is not None:
            relevant_chunks_str = relevant_chunks_content[0]
        else:
            relevant_chunks_str = []

        completion_query = self._inject_content_prompt(
            case_description=case.description,
            supporting_content=relevant_chunks_str,
        )
        q_and_a = (SystemMessage(content=self.prompt_system), UserMessage(content=completion_query))
        answer = self.completion_model.call(
            system_user_q_and_a=q_and_a,
            temperature=self.config["temperature"]
        )
        return {
            "answer": answer,
            "support_content": relevant_chunks_str,
        }

    def predict_from_dataset(self, dataset: List[Case], export_predictions: bool = True):
        """_summary_

        Args:
            dataset (List[Case]): _description_
            export_predictions (bool, optional): _description_. Defaults to True.
        """
        # Generate prediction for all the cases in the dataset
        predictions = [self.predict(case=entry) for entry in dataset]

        # Export the predictions
        if export_predictions:
            with open(f"{time()}_predictions.json", mode="w", encoding="utf8") as f:
                json.dump(predictions, f, indent=4)

    def _retrieve_supporting_content(self, query: str) -> QueryResult:
        # Embed the query
        vector_query = self.embedding_model.embed([query])

        # Retrieve relevant chunks
        relevant_chunks = self.vectordb.query(
            query_embeddings=vector_query,
            n_results=self.config["n_results"],
        )
        return relevant_chunks

    def _inject_content_prompt(
        self, case_description: str, supporting_content: List[str]
    ) -> str:
        completion_query = self.prompt_template

        # inject case description in the completion request
        if CASE_PLACEHOLDER in self.prompt_template:
            completion_query = completion_query.replace(CASE_PLACEHOLDER, case_description)
        else:
            raise ValueError("Could not find the query placeholder in the prompt template.")

        # inject supporting content in completion request
        if SUPPORTING_CONTENT_PLACEHOLDER in self.prompt_template:
            completion_query = completion_query.replace(
                SUPPORTING_CONTENT_PLACEHOLDER, "\n".join(supporting_content)
            )
        else:
            raise ValueError(
                "Could not find the supporting content placeholder in the prompt template."
            )
        return completion_query
