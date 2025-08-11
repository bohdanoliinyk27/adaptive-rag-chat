import os
import uuid
import re
import asyncio
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, quote

import streamlit as st
from langchain_community.chat_models import ChatOpenAI

from create_index import ingest_file
from rag_summary import PineconeVectorStore, rag_answer

from langchain_core.messages import HumanMessage,AIMessage
from chat_engine import ask_question

# OpenAI Agent
from chat_engine_openai import create_agent as create_openai_agent, tool_tracking as openai_tool_tracking
from agents import Runner

# LangChain Agent
from chat_engine_langchain import create_langchain_agent, tool_tracking as lc_tool_tracking


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
SUMMARY_DIR = Path("summaries")
SUMMARY_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Adaptive RAG Chat", layout="wide")


def main():
    if "files" not in st.session_state:
        st.session_state.update({
            "files": [],
            "ingested": set(),
            "namespaces": {},
            "chats": {},
            "current_chat": None,
            "histories": {},
        })


    mode = st.sidebar.radio(
        "Select engine mode",
        ["LangGraph Engine", "OpenAI Agent", "LangChain Agent"]
    )

    st.sidebar.header("File Upload")
    with st.sidebar.form("upload_form", clear_on_submit=True):
        uploaded = st.file_uploader("PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True)
        upload_button = st.form_submit_button("Upload")

    if upload_button and uploaded:
        for uf in uploaded:
            if uf.name not in st.session_state["ingested"]:
                dest = UPLOAD_DIR / uf.name
                with open(dest, "wb") as f:
                    f.write(uf.getbuffer())
                report = ingest_file(str(dest), uf.name)
                st.sidebar.success(report)

                vector_store = PineconeVectorStore(namespace=uf.name)
                llm_model = ChatOpenAI(model_name="gpt-4o", temperature=0.0)
                summary = rag_answer(
                    "Please provide a concise summary of the following document content.",
                    vector_store,
                    llm_model,
                    k=5
                )
                summary_path = SUMMARY_DIR / f"{dest.stem}.txt"
                summary_path.write_text(summary, encoding="utf-8")
                st.sidebar.info(f"Summary saved as {summary_path.relative_to('.')}")

                st.session_state["files"].append(uf)
                st.session_state["ingested"].add(uf.name)
                st.session_state["namespaces"][uf.name] = uf.name
        st.rerun()

    st.sidebar.subheader("Uploaded Files")
    for idx, uf in enumerate(st.session_state["files"]):
        c1, c2 = st.sidebar.columns([4, 1])
        c1.write(uf.name)
        if c2.button("X", key=f"delf_{idx}"):
            st.session_state["files"].pop(idx)
            st.session_state["ingested"].discard(uf.name)
            st.session_state["namespaces"].pop(uf.name, None)
            st.rerun()

    chat_col, manage_col = st.columns([3, 1])

    with manage_col:
        st.header("Chats")
        if st.button("New Chat"):
            cid = uuid.uuid4().hex[:8]
            st.session_state["chats"][cid] = f"Chat {len(st.session_state['chats']) + 1}"
            st.session_state["histories"][cid] = []
            st.session_state["current_chat"] = cid
            st.rerun()

        chat_ids = list(st.session_state["chats"].keys())
        if chat_ids:
            sel = st.radio(
                "Select Chat",
                options=chat_ids,
                index=chat_ids.index(st.session_state["current_chat"]) if st.session_state["current_chat"] in chat_ids else 0,
                format_func=lambda cid: st.session_state["chats"][cid]
            )
            if sel != st.session_state["current_chat"]:
                st.session_state["current_chat"] = sel
                st.rerun()
            if st.button("Delete Chat"):
                st.session_state["chats"].pop(sel)
                st.session_state["histories"].pop(sel, None)
                st.session_state["current_chat"] = None
                st.rerun()
        else:
            st.info("No chats created")

    if st.session_state["current_chat"] is None:
        manage_col.warning("Create or select a chat")
        return

    history = st.session_state["histories"][st.session_state["current_chat"]]

    with chat_col:
        st.header("Chat")
        for role, msg in history:
            if role == "user":
                st.markdown(f"**You:** {msg}")
            else:
                if isinstance(msg, dict):
                    resp = msg.get("answer", "")
                else:
                    resp = msg

                if isinstance(resp, str):
                    url_match = re.search(r'(https?://\S+?)(?=\s|$)', resp)
                    path_match = re.search(r'([A-Za-z]:\\\S+)', resp)

                    if url_match:
                        img_url = url_match.group(1).replace("&encoding=base64", "")
                        img_url = img_url.lstrip('<([{\'"')
                        img_url = img_url.rstrip('>)]}\'",.;:')
                        st.image(img_url)
                    elif path_match and Path(path_match.group(1)).is_file():
                        st.image(path_match.group(1))
                    else:
                        st.markdown(f"**Assistant:** {resp}")
                else:
                    st.markdown(f"**Assistant:** {resp}")

                if isinstance(msg, dict) and msg.get("trace"):
                    with st.expander("Intermediate Steps"):
                        for step in msg["trace"]:
                            st.write(f"**{step['node']}**")
                            st.json(step["output"])

                if isinstance(msg, dict) and msg.get("tools"):
                    with st.expander("Tool Calls"):
                        st.json(msg["tools"])

        question = st.text_input("Your question:", key="input")
        attached = st.multiselect(
            "Select file(s) for RAG",
            options=[f.name for f in st.session_state["files"]],
            key="attached"
        )

        if st.button("Send") and question:
            history.append(("user", question))

            if mode == "LangGraph Engine":
                result = ask_question(question, attached, history)
                history.append(("assistant", {"answer": result["answer"], "trace": result["trace"]}))

            elif mode == "OpenAI Agent":
                openai_tool_tracking.clear()
                agent = create_openai_agent(attached, history)
                resp = asyncio.run(Runner.run(agent, question))
                answer = getattr(resp, "final_output", resp)
                logs = openai_tool_tracking.copy()
                history.append(("assistant", {"answer": answer, "tools": logs}))

            else:  # LangChain Agent
                lc_tool_tracking.clear()
                agent = create_langchain_agent(attached)

                # 1) Invoke with a list of HumanMessage
                raw_response = agent.invoke({
                    "messages": [HumanMessage(content=question)]
                })

                msgs = raw_response.get("messages", [])

                answer = None
                for m in reversed(msgs):
                    if isinstance(m, AIMessage):
                        answer = m.content
                        break

                if answer is None:
                    if isinstance(raw_response, dict) and "output" in raw_response:
                        answer = raw_response["output"]
                    else:
                        answer = str(raw_response)

                logs = lc_tool_tracking.copy()
                history.append(("assistant", {"answer": answer, "tools": logs}))

            st.rerun()

    st.session_state["histories"][st.session_state["current_chat"]] = history


if __name__ == "__main__":
    main()