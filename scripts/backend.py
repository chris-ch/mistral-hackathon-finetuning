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
import starlette
from autogen_chat import AutogenChat
import uvicorn
from dotenv import load_dotenv

from classifier import Classifier
from helpers import setup_logging_levels
from rag import LegalType, RAGModel


load_dotenv()


async def send_to_client(websocket: WebSocket, autogen_chat: AutogenChat):
    """_summary_

    Args:
        autogen_chat (AutogenChat): _description_
    """
    while True:
        reply = await autogen_chat.client_receive_queue.get()
        logging.info("received reply: %s", reply)
        if reply and reply == "DO_FINISH":
            autogen_chat.client_receive_queue.task_done()
            break
        await websocket.send_text(reply)
        autogen_chat.client_receive_queue.task_done()
        await asyncio.sleep(0.05)


async def receive_from_client(websocket: WebSocket, autogen_chat: AutogenChat):
    """_summary_

    Args:
        autogen_chat (AutogenChat): _description_
    """
    try:
        while True:
            logging.info("waiting for websocket text (async)")
            data = await websocket.receive_text()
            logging.info("received websocket text (async): %s", data)
            if data and data == "DO_FINISH":
                await autogen_chat.client_receive_queue.put("DO_FINISH")
                await autogen_chat.client_sent_queue.put("DO_FINISH")
                break
            await autogen_chat.client_sent_queue.put(data)
            await asyncio.sleep(0.05)
    except starlette.websockets.WebSocketDisconnect:
        logging.warning("chat interrupted")


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


def main():
    """_summary_
    """

    setup_logging_levels()

    if "MISTRAL_API_KEY" not in os.environ:
        logging.error("MISTRAL_API_KEY environment variable is not set")
        sys.exit(1)

    active_connections: list[AutogenChat] = []
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
    async def _(websocket: WebSocket, chat_id: str):
        """_summary_

        Args:
            websocket (WebSocket): _description_
            chat_id (str): _description_
        """
        logging.info("received request for chat #%s", chat_id)
        try:
            autogen_chat = AutogenChat(chat_id=chat_id)
            
            logging.info("waiting for chat %s to connect", autogen_chat.chat_id)
            await websocket.accept()
            
            logging.info("connection accepted %s", autogen_chat.chat_id)
            active_connections.append(autogen_chat)
            
            data = await websocket.receive_text()
            logging.info("received websocket text: '%s'", data)
            
            _ = asyncio.gather(
                send_to_client(websocket, autogen_chat), receive_from_client(websocket, autogen_chat)
            )
            
            logging.info("waiting for agent dona to help clarifying: %s", data)
            await autogen_chat.clarify(data)

            last_dona_message = autogen_chat.agent_dona.last_message()["content"]
            logging.info("parsing last dona message: %s", last_dona_message)
            case_summary = extract_case_summary(last_dona_message)
            
            logging.info("case Summary: %s", case_summary)

            # classify the case summary
            case_type = classifier.predict(case_summary)
            logging.info("case Type: %s",case_type)

            reply = {
                "sender": "rachel",
                "content": f"I believe the case falls under {case_type} Law. I'm looking for relevant laws that apply...",
            }
            logging.info("sending websockt text %s", reply)
            await websocket.send_text(json.dumps(reply))
            await asyncio.sleep(0.05)

            rag_output = rag_models[LegalType.from_string(mas_to_mad[case_type])].find_sources_and_answer(case_summary, temperature=0.)

            logging.info("found relevant articles: %s", rag_output['support_content'])

            reply = {
                "sender": "rachel",
                "content": rag_output["answer"],
                "sources": [],
            }
            logging.info("sending websockt text %s", reply)
            await websocket.send_text(json.dumps(reply))
            await asyncio.sleep(0.05)
            # here add the next steps (classification ...)
            # await autogen_chat.research(data)
        finally:
            logging.info("autogen_chat %s disconnecting", autogen_chat.chat_id)
            autogen_chat.client_receive_queue.put_nowait("DO_FINISH")
            active_connections.remove(autogen_chat)

    uvicorn.run(app, host="localhost", port=8000)


if __name__ == "__main__":
    main()
