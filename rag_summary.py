from typing import Any, List
from pinecone_client import index
from langchain.schema import Document, HumanMessage

class PineconeVectorStore:
    def __init__(self, namespace: str, top_k: int = 5):
        self.namespace = namespace
        self.top_k = top_k

    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        k = k or self.top_k
        resp = index.search(
            namespace=self.namespace,
            query={
                "inputs": {"text": query},
                "top_k": k
            },
            fields=["chunk_text"]
        )
        hits = resp.get("result", {}).get("hits", [])
        return [Document(page_content=h["fields"]["chunk_text"]) for h in hits]

def rag_answer(query: str, vector_store: Any, llm: Any, k: int = 5) -> str:
    docs = vector_store.similarity_search(query, k=k)
    docs_content = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""You are an assistant that answers questions exclusively based on provided context.

Context:
{docs_content}

Instruction:
{query}

Answer:"""

    response = llm([HumanMessage(content=prompt)])
    if hasattr(response, "content"):
        return response.content
    gens = getattr(response, "generations", None)
    if gens and isinstance(gens, list) and gens and hasattr(gens[0][0], "text"):
        return gens[0][0].text
    return str(response)
