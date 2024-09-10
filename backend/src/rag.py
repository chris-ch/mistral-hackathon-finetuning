"""
"""
from enum import Enum
import logging
from typing import List, Optional, Tuple

import sys

import chromadb
from chromadb.api.types import QueryResult
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from mistralai import Mistral, SystemMessage, UserMessage

from embedding import EmbeddingModel
from prompts import CASE_PLACEHOLDER, PROMPT_SYSTEM, PROMPT_TEMPLATE, SUPPORTING_CONTENT_PLACEHOLDER


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
            temperature: impact the imagination an_summary_
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
        answer = response.choices[0].message.content
        return answer if answer is not None else "Failed to complete"


class RAGModel:
    """_summary_
    """
    def __init__(self, api_key: str,
                 legal_type: LegalType,
                 count_results: int = 5,
                 embedding_model="mistral-embed",
                 completion_model="mistral-large-latest"):
        """Model responsible for consuming the data to build a knowledge database"""
        self.prompt_system = PROMPT_SYSTEM
        self.count_results = count_results

        # Embedding model, convert natural langage to vector
        self.embedding_model = EmbeddingModel(model_deployment=embedding_model, api_key=api_key)

        logging.info("creating Vector DB persistent client for expert '%s'", legal_type.value)

        # Vector database/Search indexpredict
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
            api_key=api_key,
            model_deployment=completion_model,
        )

    def find_sources_and_answer(self, case_description: str, temperature: float = 0.):
        """
        Execute all the steps of the RAG logic:
            1. embed the query
            2. retrieve the supporting content
            3. update the prompt with the information
            4. return the completion model reponse
        """
        logging.info("starting predict phase")
        relevant_chunks = self._retrieve_relevant_documents(case_description)

        # convert relevant chunks to a list of string
        relevant_chunks_content = relevant_chunks["documents"]
        if relevant_chunks_content is not None:
            relevant_chunks_str = relevant_chunks_content[0]
        else:
            relevant_chunks_str = ""

        completion_query = self._inject_content_prompt(
            case_description=case_description,
            supporting_content=relevant_chunks_str,
        )
        q_and_a = (SystemMessage(content=self.prompt_system), UserMessage(content=completion_query))
        answer = self.completion_model.call(
            system_user_q_and_a=q_and_a,
            temperature=temperature
        )
        return {
            "answer": answer,
            "support_content": relevant_chunks_str,
        }

    def _retrieve_relevant_documents(self, query: str) -> QueryResult:
        # Embed the query
        vector_query = self.embedding_model.embed([query])

        # Retrieve relevant chunks
        relevant_chunks = self.vectordb.query(query_embeddings=vector_query,
                                              n_results=self.count_results)
        return relevant_chunks

    def _inject_content_prompt(
        self, case_description: str, supporting_content: List[str]
    ) -> str:

        # inject case description in the completion request
        completion_query = PROMPT_TEMPLATE.replace(CASE_PLACEHOLDER, case_description)

        # inject supporting content in completion request
        completion_query = completion_query.replace(
            SUPPORTING_CONTENT_PLACEHOLDER, "\n".join(supporting_content)
        )
        return completion_query
