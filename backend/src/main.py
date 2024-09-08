"""
"""
import re
import json
import os

import asyncio
from fastapi import FastAPI, WebSocket
from autogen_chat import AutogenChat
import uvicorn
from dotenv import load_dotenv

from classifier import Classifier
from prompts import PROMPT_SYSTEM, PROMPT_TEMPLATE
from rag import LegalType, RAGModel
from utils import Case, LawDomain

load_dotenv()

classifier = Classifier.from_pretrained("data/classifier_tfidflgbm")

config = {
    # rag knowledge path
    "knowledge_folder": "data/federal_law_short",
    # rag prompts
    "prompt_system": PROMPT_SYSTEM,
    "prompt_template": PROMPT_TEMPLATE,
    # splitter
    "chunk_size": 2000,
    "chunk_overlap": 100,
    # embedding model
    "embedding_provider": "mistral",
    "embedding_model_deployment": "mistral-embed",
    "embedding_api_key": os.environ.get("MISTRAL_API_KEY"),
    "embedding_api_version": "",
    "embedding_endpoint": "",
    # retrieval parameters
    "n_results": 5,
    # completion model
    "completion_provider": "mistral",
    "completion_model_deployment": "mistral-large-latest",
    "completion_api_key": os.environ.get("MISTRAL_API_KEY"),
    "completion_api_version": "",
    "completion_endpoint": "",
    "temperature": 0,
}

print(f'using Mistral API Key {os.environ.get("MISTRAL_API_KEY")}')

mas_to_mad = {
    "Civil": LegalType.PRIVATE.value,
    "Criminal": LegalType.CRIMINAL.value,
    "Public": LegalType.PUBLIC.value,
}

rag_models = {}
for name in mas_to_mad.values():
    print(f"creating RAG model {name}")
    rag_models[name] = RAGModel(
        expert_name=name,
        config=config
    )

app = FastAPI()
app.autogen_chat = {}


class ConnectionManager:
    def __init__(self):
        print("Connection Manager activated")
        self.active_connections: list[AutogenChat] = []

    async def connect(self, autogen_chat: AutogenChat):
        print(f"waiting for chat {autogen_chat.chat_id} to connect")
        await autogen_chat.websocket.accept()
        print(f"connection accepted {autogen_chat.chat_id}")
        self.active_connections.append(autogen_chat)

    async def disconnect(self, autogen_chat: AutogenChat):
        autogen_chat.client_receive_queue.put_nowait("DO_FINISH")
        print(f"autogen_chat {autogen_chat.chat_id} disconnected")
        self.active_connections.remove(autogen_chat)


manager = ConnectionManager()


async def send_to_client(autogen_chat: AutogenChat):
    while True:
        reply = await autogen_chat.client_receive_queue.get()
        if reply and reply == "DO_FINISH":
            autogen_chat.client_receive_queue.task_done()
            break
        await autogen_chat.websocket.send_text(reply)
        autogen_chat.client_receive_queue.task_done()
        await asyncio.sleep(0.05)


async def receive_from_client(autogen_chat: AutogenChat):
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
    # Use regex to find the text starting with "Case Summary:" and ending before the next paragraph
    match = re.search(r"Case Summary:(.*?)(?:\n\n|$)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


@app.websocket("/ws/{chat_id}")
async def websocket_endpoint(websocket: WebSocket, chat_id: str):
    print(f"received request for chat #{chat_id}")
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
        print(f"Case Summary: {case_summary}")

        # classify the case summary
        case_type = classifier.predict(case_summary)
        print(f"Case Type: {case_type}")

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
        rag_output = rag_models[mas_to_mad[case_type]].predict(legal_case)

        print(f"Relevant articles: {rag_output['support_content']}")

        reply = {
            "sender": "rachel",
            "content": rag_output["answer"],
            "sources": [],
        }
        await autogen_chat.websocket.send_text(json.dumps(reply))
        await asyncio.sleep(0.05)
        # here add the next steps (classification ...)
        # await autogen_chat.research(data)
    except Exception as e:
        print("ERROR", str(e))
    finally:
        try:
            await manager.disconnect(autogen_chat)
        except:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
