import os
import uuid
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import openai
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import StructuredTool, Tool
from langchain.schema import HumanMessage

from pydantic import BaseModel
from typing import List

class RAGToolInput(BaseModel):
    query: str
    namespaces: List[str]

class CiteToolInput(BaseModel):
    answer: str
    namespaces: List[str]

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "developer-quickstart-py")
index = pc.Index(INDEX_NAME)

IMAGE_DIR = Path(__file__).parent / "generated_images"
IMAGE_DIR.mkdir(exist_ok=True)

tool_tracking: List[Dict[str, Any]] = []

llm_model = ChatOpenAI(model_name="gpt-4o", temperature=0.0)

def llm_chat(query: str) -> str:
    response = llm_model([HumanMessage(content=query)])
    return response.content


def generate_rag(query: str, namespaces: List[str]) -> str:
    """
    Input: full users query, attached namespaces
    Output: Answer based on the users query

    Retrieve relevant document chunks from Pinecone and answer using GPT-4o.
    Always pass full users query to this tool
    """
    all_chunks: List[str] = []
    for ns in namespaces:
        resp = index.search(
            namespace=ns,
            query={"inputs": {"text": query}, "top_k": 5},
            fields=["chunk_text"]
        )
        hits = resp.get("result", {}).get("hits", [])
        all_chunks.extend(h["fields"]["chunk_text"] for h in hits)
    context = "\n\n---\n\n".join(all_chunks)

    prompt = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    answer = llm_chat(prompt)

    tool_tracking.append({
        "tool": "generate_rag",
        "user_prompt": query,
        "chunks": context,
        "response": answer
    })
    return answer


def generate_image(query: str) -> str:
    """
    Input: users query
    Output: Path to file with image
    Generate an image via DALL·E and save it locally.
    If returns a file path, include it exactly as given in your reply—do not convert it to Markdown.
    """
    client = openai.OpenAI()
    prompt = query.strip() + " do not crop the image"
    resp = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="auto",
        quality="auto",
        n=1,
    )
    img_data = resp.data[0].b64_json
    img_bytes = base64.b64decode(img_data)

    filename = f"image_{datetime.now():%Y%m%d%H%M%S}_{uuid.uuid4().hex[:8]}.png"
    filepath = IMAGE_DIR / filename
    with open(filepath, "wb") as f:
        f.write(img_bytes)

    result = str(filepath.resolve())
    tool_tracking.append({"tool": "generate_image", "response": result})
    return result


def generate_chart(query: str) -> str:
    """
    Build a QuickChart.io URL for a chart based on the user's instructions.
    """
    prompt = (
        "Build GET URL to https://quickchart.io/chart with Chart.js config; return only the URL.\n"
        f"User request: {query}"
    )
    url = llm_chat(prompt)
    tool_tracking.append({"tool": "generate_chart", "response": url})
    return url


def add_citations(answer: str, namespaces: List[str]) -> str:
    """
    Input: full users query, attached namespaces

    Add in-text citations to the given answer using file summaries.
    After generate_rag after retrieving information, always invoke the add_citations tool to include citations in your answer.
    """
    summary_dir = Path("summaries")
    summaries: List[str] = []
    for ns in namespaces:
        sf = summary_dir / f"{Path(ns).stem}.txt"
        if sf.exists():
            summaries.append(f"{Path(ns).name}: {sf.read_text(encoding='utf-8')}")
    prompt = (
        "Insert [filename] citations into each paragraph of the answer.|n|n"
        "Files:\n" + "\n".join(namespaces) + "\n\n"
        "Summaries:\n" + "\n".join(summaries) + "\n\n"
        "Answer:\n" + answer
    )
    annotated = llm_chat(prompt)
    tool_tracking.append({"tool": "add_citations", "response": annotated})
    return annotated

rag_tool = StructuredTool.from_function(
    generate_rag,
    name="generate_rag",
    description=(
        "Retrieve context chunks from Pinecone and answer via RAG. "
        "Arguments: query (str), namespaces (List[str])"
    ),
    args_schema=RAGToolInput
)


cite_tool = StructuredTool.from_function(
    add_citations,
    name="add_citations",
    description=(
        "Always after generate_rag; add in-text citations. Always pass answer and namespaces as input"
    ),
    args_schema=CiteToolInput
)
image_tool = Tool.from_function(generate_image, name="generate_image", description="Generate DALL·E image; return local path exactly.")
chart_tool = Tool.from_function(generate_chart, name="generate_chart", description="Generate QuickChart.io URL.")


def create_langchain_agent(namespaces: List[str]):
    tools = [image_tool, chart_tool, cite_tool]
    instructions = (
        "You are a helpful assistant that can answer questions, "
        "generate images and charts, and add in-text citations when needed. "
        "Invoke the appropriate tool based on the user’s request."
        "Call generate_image tool only when user asks for a picture or image"
        "If the generate_image tool returns a file path, include it exactly as given in your reply—do not convert it to Markdown."
    )
    if namespaces:
        tools.insert(0, rag_tool)
        instructions += (
            " Since files are provided, always invoke the generate_rag tool first, "
            "and after retrieving information, always invoke the add_citations tool to include citations in your answer."
            "Always pass full users query and namespaces in input of the generate_rag tool and add_citations tool"
            f"namespaces attached: {namespaces}"
        )




    agent = create_react_agent(
        model=llm_model,
        tools=tools,
        prompt=instructions,
    )
    return agent