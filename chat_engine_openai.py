import os
import uuid
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

import openai
from dotenv import load_dotenv
from agents import Agent, function_tool
from pinecone_client import index

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Directory for locally saved images
IMAGE_DIR = Path(__file__).parent / "generated_images"
IMAGE_DIR.mkdir(exist_ok=True)

# Global tracking log for tool invocations
tool_tracking: List[Dict[str, Any]] = []

# -----------------
# LLM Chat Wrapper
# -----------------
def llm_chat(query: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

# -----------------
# RAG Tool
# -----------------
@function_tool
def generate_rag(query: str, namespaces: List[str]) -> str:
    """
    Input: full users query, attached namespaces
    Output: Answer based on the users query

    Retrieve relevant document chunks from Pinecone and answer using GPT-4o.
    Always pass full users query to this tool
    """
    tool_name = "generate_rag"
    try:
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

        prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        annotated = resp.choices[0].message.content.strip()
        tool_tracking.append({"user_prompt":query, "tool": tool_name, "chunks": context,"response": annotated})
        return annotated
    except Exception as e:
        tool_tracking.append({"tool": tool_name, "error": str(e)})
        raise

# -----------------
# Image Generation Tool
# -----------------
@function_tool
def generate_image(query: str) -> str:
    """
    Generate an image via DALL·E and save it locally.
    """
    tool_name = "generate_image"
    prompt = query.strip() + " do not crop the image"
    try:
        client = openai.OpenAI()
        resp = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="auto",
            quality="auto",
            n=1,
        )
        response = resp
        img_data = resp.data[0].b64_json
        img_bytes = base64.b64decode(img_data)
        filename = f"image_{datetime.now():%Y%m%d%H%M%S}_{uuid.uuid4().hex[:8]}.png"
        filepath = IMAGE_DIR / filename
        with open(filepath, "wb") as f:
            f.write(img_bytes)
        result = str(filepath.resolve())
        tool_tracking.append({"tool": tool_name, "response": result})
        return result
    except Exception as e:
        tool_tracking.append({"tool": tool_name, "error": str(e)})
        raise

# -----------------
# Chart Generation Tool
# -----------------
@function_tool
def generate_chart(query: str) -> str:
    """
    Build a QuickChart.io GET URL based on Chart.js config.
    """
    tool_name = "generate_chart"
    prompt = (
        "Build GET URL to https://quickchart.io/chart with Chart.js config; return only the URL."
        "Example of correct chart format https://quickchart.io/chart?c=%7B%22type%22%3A%22bar%22%2C%22data%22%3A%7B%22labels%22%3A%5B%22Red%22%2C%22Blue%22%2C%22Yellow%22%2C%22Green%22%2C%22Purple%22%2C%22Orange%22%5D%2C%22datasets%22%3A%5B%7B%22label%22%3A%22%23%20of%20Votes%22%2C%22data%22%3A%5B12%2C19%2C3%2C5%2C2%2C3%5D%2C%22backgroundColor%22%3A%5B%22rgba(255%2C%2099%2C%20132%2C%200.2)%22%2C%22rgba(54%2C%20162%2C%20235%2C%200.2)%22%2C%22rgba(255%2C%20206%2C%2086%2C%200.2)%22%2C%22rgba(75%2C%20192%2C%20192%2C%200.2)%22%2C%22rgba(153%2C%20102%2C%20255%2C%200.2)%22%2C%22rgba(255%2C%20159%2C%2064%2C%200.2)%22%5D%2C%22borderColor%22%3A%5B%22rgba(255%2C%2099%2C%20132%2C%201)%22%2C%22rgba(54%2C%20162%2C%20235%2C%201)%22%2C%22rgba(255%2C%20206%2C%2086%2C%201)%22%2C%22rgba(75%2C%20192%2C%20192%2C%201)%22%2C%22rgba(153%2C%20102%2C%20255%2C%201)%22%2C%22rgba(255%2C%20159%2C%2064%2C%201)%22%5D%2C%22borderWidth%22%3A1%7D%5D%7D%2C%22options%22%3A%7B%22scales%22%3A%7B%22y%22%3A%7B%22beginAtZero%22%3Atrue%7D%7D%7D%7D"
        f"\nUser request: {query}"
    )
    try:
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        url = resp.choices[0].message.content.strip()
        tool_tracking.append({"tool": tool_name, "response": url})
        return url
    except Exception as e:
        tool_tracking.append({"tool": tool_name, "error": str(e)})
        raise

# -----------------
# In-Text Citation Tool
# -----------------
@function_tool
def add_citations(answer: str, namespaces: List[str]) -> str:
    """
    Add in-text citations to each paragraph of an existing answer based on file summaries.
    """
    tool_name = "add_citations"
    try:
        summary_dir = Path("summaries")
        summaries: List[str] = []
        for ns in namespaces:
            summary_file = summary_dir / f"{Path(ns).stem}.txt"
            if summary_file.exists():
                content = summary_file.read_text(encoding="utf-8")
                summaries.append(f"{Path(ns).name}: {content}")
        prompt = (
            "Insert [filename] citations into each paragraph of the answer.\n\n"
            "Files:\n" + "\n".join(namespaces) + "\n\n"
            "Summaries:\n" + "\n".join(summaries) + "\n\n"
            "Answer:\n" + answer
        )
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        annotated = resp.choices[0].message.content.strip()
        tool_tracking.append({"tool": tool_name, "response": annotated})
        return annotated
    except Exception as e:
        tool_tracking.append({"tool": tool_name, "error": str(e)})
        raise

# -----------------
# Agent Factory
# -----------------
def create_agent(
    namespaces: List[str],
    history: List[Tuple[str, str]]
) -> Agent:

    """
    Create an Agent with appropriate tools.
    Only include RAG tool if namespaces is non-empty.
    """
    tools = [generate_image, generate_chart, add_citations]

    # Core instructions for the agent
    instructions = (
        "You are a helpful assistant that can answer questions, "
        "generate images and charts, and add in-text citations when needed. "
        "Invoke the appropriate tool based on the user’s request."
        "If the generate_image tool returns a file path, include it exactly as given in your reply—do not convert it to Markdown."
    )


    if namespaces:
        tools.insert(0, generate_rag)
        instructions += (
            " Since files are provided, always invoke the generate_rag tool first, "
            "and after retrieving information, always invoke the add_citations tool to include citations in your answer."
            f"namespaces attached: {namespaces}"
        )
    if history:
        history_text = "\n".join(
            f"{'You' if role == 'user' else 'Assistant'}: {msg}"
            for role, msg in history
        )
        instructions += "\n\nChat History:\n" + history_text
    return Agent(name="DocAgent", tools=tools, instructions=instructions, model="gpt-4o")
