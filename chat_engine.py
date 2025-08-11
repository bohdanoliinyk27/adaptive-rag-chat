import os
import uuid
import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, TypedDict, Tuple
from copy import deepcopy
from dotenv import load_dotenv

from pinecone_client import index
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph, START, END
from openai import OpenAI

load_dotenv()

IMAGE_DIR = Path(__file__).parent / "generated_images"
IMAGE_DIR.mkdir(exist_ok=True)

# Primary LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0)


def llm_chat(query: str) -> str:
    """Simple LLM chat wrapper."""
    return llm([HumanMessage(content=query)]).content


def image_generation(data: Any) -> str:
    """
    Call OpenAI DALL·E to generate an image from `data` (str or dict with 'text').
    Saves locally and returns the file path.
    """
    if isinstance(data, dict):
        prompt_text = data.get('text', "")
    elif isinstance(data, str):
        prompt_text = data
    else:
        raise TypeError(f"{type(data)}; Expected str or dict")

    prompt = prompt_text.strip() + " do not crop the image"
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="auto",
        quality="auto",
        n=1,
    )
    img_data = resp.data[0].b64_json
    img_bytes = base64.b64decode(img_data)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f"image_{timestamp}_{uuid.uuid4().hex[:8]}.png"
    filepath = IMAGE_DIR / filename
    with open(filepath, "wb") as f:
        f.write(img_bytes)
    return str(filepath.resolve())


# Trace wrapper to record node inputs/outputs
def wrap_node(fn, name: str):
    def wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
        new_state = fn(deepcopy(state))
        trace = deepcopy(state.get("trace", []))
        output: Dict[str, Any] = {}
        for key in ("needs_image", "needs_chart", "use_rag", "chunks", "kept_summaries", "answer", "question"):
            if key in new_state and new_state.get(key) != state.get(key):
                output[key] = new_state[key]
        trace.append({"node": name, "output": output})
        new_state["trace"] = trace
        return new_state
    return wrapped


# --- Chart generation nodes ---
def needs_chart(state: TypedDict) -> TypedDict:
    prompt = f"Decide if the user wants a chart not image!. Answer 'yes' or 'no'.\nUser request: {state['question']}"
    resp = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)([HumanMessage(content=prompt)]).content.strip().lower()
    return {**state, "needs_chart": resp.startswith("yes")}


def generate_chart(state: TypedDict) -> TypedDict:
    instructions = "Build GET URL to https://quickchart.io/chart with Chart.js config; return only the URL."
    prompt = instructions + f"\nUser request: {state['question']}"
    url = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)([HumanMessage(content=prompt)]).content.strip()
    return {**state, "answer": url}


# --- Image generation nodes ---
def needs_image(state: TypedDict) -> TypedDict:
    prompt = f"Decide if the user wants an image not chart. Answer 'yes' or 'no'.\nUser request: {state['question']}"
    resp = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)([HumanMessage(content=prompt)]).content.strip().lower()
    return {**state, "needs_image": resp.startswith("yes")}


def generate_image(state: TypedDict) -> TypedDict:
    link = image_generation(state['question'])
    return {**state, "answer": link}


# --- RAG logic nodes ---
class GraphState(TypedDict, total=False):
    question: str
    attached: bool
    namespaces: List[str]
    history: List[Tuple[str, str]]
    needs_chart: bool
    needs_image: bool
    use_rag: bool
    chunks: List[str]
    kept_summaries: List[str]
    answer: str
    trace: List[Dict[str, Any]]
    attempts: int


def check_rag_usage(state: GraphState) -> GraphState:
    summary_dir = Path("summaries")
    summaries = []
    for ns in state.get("namespaces", []):
        summary_file = summary_dir / f"{Path(ns).stem}.txt"
        if summary_file.exists():
            content = summary_file.read_text(encoding="utf-8")
            summaries.append(f"{Path(ns).name}: {content}")
    prompt = (
        f"User query: {state['question']}\n"
        f"Summaries: {'; '.join(summaries)}\n?"
        "Is this question related to this file (or files)? "
        "Can it be answered using only the information in the file(s)? "
        "If user mentioned file in the query always answer yes"
        "If the file summaries do not contain relevant information to the user’s query, answer \"no.\" yes/no"
    )
    resp = ChatOpenAI(model_name="gpt-4o", temperature=0.0)([HumanMessage(content=prompt)]).content.strip().lower()
    return {**state, "use_rag": resp.startswith("yes"), "chunks": [], "kept_summaries": summaries}


def retrieve(state: GraphState) -> GraphState:
    all_chunks: List[str] = []
    for ns in state.get("namespaces", []):
        resp = index.search(
            namespace=ns,
            query={"inputs": {"text": state['question']}, "top_k": 5},
            fields=["chunk_text"]
        )
        hits = resp.get("result", {}).get("hits", [])
        all_chunks.extend(h["fields"]["chunk_text"] for h in hits)
    return {**state, "chunks": all_chunks}


def grade_documents(state: GraphState) -> GraphState:
    # 1) Filter chunks as before
    kept = []
    grader = ChatOpenAI(model_name="gpt-4o", temperature=0.0)
    for chunk in state.get("chunks", []):
        prompt = (
            f"Text: {chunk}\n"
            f"Question: {state['question']}\n"
            "Help answer? yes/no."
        )
        reply = grader([HumanMessage(content=prompt)]).content.strip().lower()
        if reply.startswith("yes"):
            kept.append(chunk)

    # 2) Load summaries from files named after each namespace
    summary_dir = Path("summaries")
    summaries = []
    for ns in state.get("namespaces", []):
        summary_file = summary_dir / f"{Path(ns).stem}.txt"
        if summary_file.exists():
            content = summary_file.read_text(encoding="utf-8")
            summaries.append(f"{Path(ns).name}: {content}")

    # 3) Grade each loaded summary
    filtered = []
    for summ in summaries:
        prompt = (
            f"Summary: {summ}\n"
            f"Question: {state['question']}\n"
            "Is this question related to this Summary? "
            "Can it be answered using only the information in the Summary? "
            "If user mentioned file in the query always answer yes "
            "If the file summaries do not contain relevant information to the user’s query, answer \"no.\" yes/no"
        )
        reply = grader([HumanMessage(content=prompt)]).content.strip().lower()
        if reply.startswith("yes"):
            filtered.append(summ)

    # 4) Return new state with filtered chunks & summaries
    return {
        **state,
        "chunks": kept,
        "kept_summaries": filtered
    }


def transform_query(state: GraphState) -> GraphState:
    prompt = f"Rewrite query given summaries {'; '.join(state.get('kept_summaries', []))}: {state['question']}"
    new_q = ChatOpenAI(model_name="gpt-4o", temperature=0.0)([HumanMessage(content=prompt)]).content.strip()
    return {**state, "question": new_q, "attempts": state.get("attempts", 0) + 1}


def generate_final(state: GraphState) -> GraphState:
    convo = "\n".join(f"{r.title()}: {m}" for r, m in state.get("history", []))
    context = "\n".join(state.get('chunks', []) + state.get('kept_summaries', []))
    prompt = f"History: {convo}\nContext: {context}\nQuestion: {state['question']}\nAnswer:"
    return {**state, "answer": llm_chat(prompt)}


def llm_generate(state: GraphState) -> GraphState:
    convo = "\n".join(f"{role.title()}: {msg}" for role, msg in state.get("history", []))
    prompt = f"History:\n{convo}\n\nQuestion: {state['question']}\nAnswer:"
    return {**state, "answer": llm_chat(prompt)}


def add_intext_citations(state: GraphState) -> GraphState:
    previous_answer = state.get("answer", "")
    summaries = state.get("kept_summaries", [])
    files = state.get("namespaces", [])

    prompt = (
        "You are an assistant whose task is to add in-text citations to an already generated answer. "
        "For each paragraph in the answer, insert a citation in square brackets indicating the source file. "
        "Decide which file each paragraph is drawn from by using the provided summaries.\n\n"
        "Currently attached files:\n" +
        "\n".join(f"- {f}" for f in files) +
        "\n\n"
        "File summaries:\n" +
        "\n".join(summaries) +
        "\n\n"
        "Answer to annotate:\n" +
        previous_answer +
        "\n\n"
        "Rewrite the answer, preserving content but adding [filename.ext] after each paragraph."
    )

    annotated = ChatOpenAI(model_name="gpt-4o", temperature=0.0)(
        [HumanMessage(content=prompt)]
    ).content.strip()

    return {**state, "answer": annotated}


def decide_after_grade(state: GraphState) -> str:
    if state.get("chunks"):
        return "generate_final"
    if state.get("kept_summaries"):
        return "transform_query"
    return "llm_generate"


def decide_route(state: GraphState) -> str:
    if state.get("needs_image"):
        return "generate_image"
    if state.get("needs_chart"):
        return "generate_chart"
    if state.get("attached"):
        return "check_rag"
    return "llm_generate"

# Build and compile the graph
graph = StateGraph(GraphState)
for name, fn in [
    ("needs_image", needs_image),
    ("generate_image", generate_image),
    ("needs_chart", needs_chart),
    ("generate_chart", generate_chart),
    ("check_rag", check_rag_usage),
    ("retrieve", retrieve),
    ("grade_documents", grade_documents),
    ("transform_query", transform_query),
    ("generate_final", generate_final),
    ("llm_generate", llm_generate),
    ("add_citations", add_intext_citations),
]:
    graph.add_node(name, wrap_node(fn, name))

# Routing edges
graph.add_edge(START, "needs_image")
graph.add_conditional_edges(
    "needs_image",
    lambda s: "generate_image" if s.get("needs_image") else "needs_chart",
    {"generate_image": "generate_image", "needs_chart": "needs_chart"}
)

graph.add_conditional_edges(
    "needs_chart",
    decide_route,
    {
        "generate_image": "generate_image",
        "generate_chart": "generate_chart",
        "check_rag":       "check_rag",
        "llm_generate":     "llm_generate",
    }
)

# Only proceed to retrieve if use_rag == True
graph.add_conditional_edges(
    "check_rag",
    lambda s: "retrieve" if s.get("use_rag") else "llm_generate",
    {"retrieve": "retrieve", "llm_generate": "llm_generate"}
)

# RAG chain continuation
graph.add_edge("retrieve", "grade_documents")
graph.add_conditional_edges(
    "grade_documents",
    decide_after_grade,
    {"transform_query": "transform_query", "generate_final": "generate_final", "llm_generate": "llm_generate"}
)
graph.add_edge("transform_query", "retrieve")
graph.add_edge("generate_final", "add_citations")
graph.add_edge("add_citations", END)

# Terminal edges
graph.add_edge("generate_image", END)
graph.add_edge("generate_chart", END)

compiled_graph = graph.compile()

def ask_question(
    query: str,
    attached: List[str],
    history: List[Tuple[str, str]] = None
) -> Dict[str, Any]:
    state: GraphState = {
        "question": query,
        "attached": bool(attached),
        "namespaces": attached or [],
        "attempts": 0,
    }
    if history is not None:
        state["history"] = history
    state["trace"] = []
    out = compiled_graph.invoke(state)
    return {"answer": out.get("answer", ""), "trace": out.get("trace", [])}
