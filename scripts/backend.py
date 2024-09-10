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
from typing import Dict
from fastapi import FastAPI, WebSocket
from autogen_chat import AutogenChat
import uvicorn
from dotenv import load_dotenv

from classifier import Classifier
from helpers import setup_logging_levels
from rag import LegalType, RAGModel


load_dotenv()


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
    logging.info("waiting for client requests")
    while True:
        data = await autogen_chat.websocket.receive_text()
        logging.info("received request from client: %s", data)
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

app = FastAPI()
app.autogen_chat = {}


def main():
    """_summary_
    """

    setup_logging_levels()

    if "MISTRAL_API_KEY" not in os.environ:
        logging.error("MISTRAL_API_KEY environment variable is not set")
        sys.exit(1)

    manager = ConnectionManager()
    classifier = Classifier.from_pretrained("data/classifier_tfidflgbm")
    mas_to_mad = {
        "Civil": LegalType.PRIVATE,
        "Criminal": LegalType.CRIMINAL,
        "Public": LegalType.PUBLIC,
    }
    rag_models: Dict[LegalType, RAGModel] = {}
    for lt in mas_to_mad.values():
        logging.info("creating RAG model %s", lt.value)
        rag_models[lt] = RAGModel(os.environ.get("MISTRAL_API_KEY"), legal_type=lt)

    @app.websocket("/ws/{chat_id}")
    async def websocket_endpoint(websocket: WebSocket, chat_id: str):
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

            rag_output = rag_models[LegalType.from_string(mas_to_mad[case_type])].predict(case_summary, temperature=0.)

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

    uvicorn.run(app, host="localhost", port=8000)


if __name__ == "__main__":
    main()
