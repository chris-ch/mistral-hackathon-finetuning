"""_summary_

    Returns:
        _type_: _description_
"""
import logging
import re
import json
import os

import asyncio
import sys
from fastapi import FastAPI, WebSocket
from autogen_chat import AutogenChat
import uvicorn
from dotenv import load_dotenv

from classifier import Classifier
from prompts import PROMPT_SYSTEM, PROMPT_TEMPLATE
from rag import LegalType, RAGModel
from utils import Case, LawDomain

load_dotenv()

config = {
    # rag prompts
    "prompt_system": PROMPT_SYSTEM,
    "prompt_template": PROMPT_TEMPLATE,
    # embedding model
    "embedding_provider": "mistral",
    "embedding_model_deployment": "mistral-embed",
    "embedding_api_key": os.environ.get("MISTRAL_API_KEY"),
    "embedding_api_version": "",
    "embedding_endpoint": "",
    # retrieval parameters
    "n_results": 5,
    # completion model
    "completion_model_deployment": "mistral-large-latest",
    "completion_api_key": os.environ.get("MISTRAL_API_KEY"),
    "completion_api_version": "",
    "completion_endpoint": "",
    "temperature": 0,
}


class ConnectionManager:
    """_summary_
    """
    def __init__(self):
        logging.info("Connection Manager activated")
        self.active_connections: list[AutogenChat] = []

    async def connect(self, autogen_chat: AutogenChat):
        """_summary_

        Args:
            autogen_chat (AutogenChat): _description_
        """
        logging.info("waiting for chat %s to connect", autogen_chat.chat_id)
        await autogen_chat.websocket.accept()
        logging.info("connection accepted %s", autogen_chat.chat_id)
        self.active_connections.append(autogen_chat)

    async def disconnect(self, autogen_chat: AutogenChat):
        """_summary_

        Args:
            autogen_chat (AutogenChat): _description_
        """
        autogen_chat.client_receive_queue.put_nowait("DO_FINISH")
        logging.info("autogen_chat %s disconnected", autogen_chat.chat_id)
        self.active_connections.remove(autogen_chat)


async def send_to_client(autogen_chat: AutogenChat):
    """_summary_

    Args:
        autogen_chat (AutogenChat): _description_
    """
    while True:
        reply = await autogen_chat.client_receive_queue.get()
        if reply and reply == "DO_FINISH":
            autogen_chat.client_receive_queue.task_done()
            break
        await autogen_chat.websocket.send_text(reply)
        autogen_chat.client_receive_queue.task_done()
        await asyncio.sleep(0.05)


async def receive_from_client(autogen_chat: AutogenChat):
    """_summary_

    Args:
        autogen_chat (AutogenChat): _description_
    """
    while True:
        data = await autogen_chat.websocket.receive_text()
        if data and data == "DO_FINISH":
            await autogen_chat.client_receive_queue.put("DO_FINISH")
            await autogen_chat.client_sent_queue.put("DO_FINISH")
            break
        await autogen_chat.client_sent_queue.put(data)
        await asyncio.sleep(0.05)


# Define a function to extract the case summary
def extract_case_summary(text):
    """_summary_

    Args:
        text (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Use regex to find the text starting with "Case Summary:" and ending before the next paragraph
    match = re.search(r"Case Summary:(.*?)(?:\n\n|$)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


try:
    mistral_api_key = os.environ["MISTRAL_API_KEY"]
except KeyError:
    logging.error("MISTRAL_API_KEY environment variable is not set")
    sys.exit(1)


app = FastAPI()
app.autogen_chat = {}

mas_to_mad = {
    "Civil": LegalType.PRIVATE,
    "Criminal": LegalType.CRIMINAL,
    "Public": LegalType.PUBLIC,
}

rag_models = {}
for lt in mas_to_mad.values():
    logging.info("creating RAG model %s", lt.value)
    rag_models[lt] = RAGModel(legal_type=lt, config=config)


mgr = ConnectionManager()
clsfr = Classifier.from_pretrained("data/classifier_tfidflgbm")

@app.websocket("/ws/{chat_id}")
async def websocket_endpoint(websocket: WebSocket, chat_id: str, manager=mgr, classifier=clsfr):
    """_summary_

    Args:
        websocket (WebSocket): _description_
        chat_id (str): _description_
    """
    logging.info("received request for chat #%s", chat_id)
    try:
        autogen_chat = AutogenChat(chat_id=chat_id, websocket=websocket)
        await manager.connect(autogen_chat)
        data = await autogen_chat.websocket.receive_text()
        _ = asyncio.gather(
            send_to_client(autogen_chat), receive_from_client(autogen_chat)
        )
        await autogen_chat.clarify(data)

        last_dona_message = autogen_chat.agent_dona.last_message()["content"]
        case_summary = extract_case_summary(last_dona_message)
        logging.info("Case Summary: %s", case_summary)

        # classify the case summary
        case_type = classifier.predict(case_summary)
        logging.info("Case Type: %s",case_type)

        reply = {
            "sender": "rachel",
            "content": f"I believe the case falls under {case_type} Law. I'm looking for relevant laws that apply...",
        }
        await autogen_chat.websocket.send_text(json.dumps(reply))
        await asyncio.sleep(0.05)

        legal_case = Case(
            description=case_summary,
            related_articles=[],
            outcome=True,
            domain=LawDomain.CRIMINAL,
        )
        rag_output = rag_models[mas_to_mad[LegalType.from_string(case_type)]].predict(legal_case)

        logging.info("Relevant articles: %s", rag_output['support_content'])

        reply = {
            "sender": "rachel",
            "content": rag_output["answer"],
            "sources": [],
        }
        await autogen_chat.websocket.send_text(json.dumps(reply))
        await asyncio.sleep(0.05)
        # here add the next steps (classification ...)
        # await autogen_chat.research(data)
    finally:
        await manager.disconnect(autogen_chat)


def main():
    """_summary_
    """
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
