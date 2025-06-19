from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from transformers import AutoModel
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import List, TypedDict

# Custom embedding model
class HFEmbeddingsWrapper:
    def __init__(self):
        self.my_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
    def embed_documents(self, texts):
        return self.my_model.encode(texts, task="text-matching")
    def embed_query(self, text):
        return self.embed_documents([text])[0]

# Build the main response graph
def build_graph():
    doc = Document('رشد.docx')  # Word file used as knowledge source
    input_data = "\n".join([x.text for x in doc.paragraphs if len(x.text) > 0])

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    input_data = text_splitter.create_documents([input_data])
    all_splits = text_splitter.split_documents(input_data)

    # Create in-memory vector store
    vector_store = InMemoryVectorStore(HFEmbeddingsWrapper())
    _ = vector_store.add_documents(documents=all_splits)

    # Prompt template for the language model
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "تو یک دستیار فارسی هستی. لطفا بر اساس context ارائه شده یک پاسخ متناسب با سوال ارائه کن "
                "و فقط پاسخ را بنویس و اگر نتونستی پاسخ مناسبی پیدا کنی و یا context خالی بود فقط بگو 'نمیدونم '.",
            ),
            (
                "human",
                "سوال: {question}"
                "\n"
                "context: {context}"
                "\n"
                "پاسخ:"
            ),
        ]
    )

    # Define the state structure for the graph
    class State(TypedDict):
        question: str
        context: List
        answer: str

    # Document retrieval step
    def retrieve(state: State):
        retrieved_docs = [x[0] for x in vector_store.similarity_search_with_score(query=state["question"], k=3) if x[1] >= 0.3]
        return {"context": retrieved_docs[:3]}

    # Response generation step using LLM
    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content.strip() for doc in state["context"]) or "خالی"
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        llm = ChatOllama(model="dorna-llama3-8b-instruct.Q8_0.gguf:latest", temperature=0)
        response = llm.invoke(messages)
        return {"answer": response.content}

    # Build the full graph
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()
