import os

# ---- ENV FIXES ----
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import hashlib
import tempfile
import warnings
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks import StreamlitCallbackHandler

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ---- SETUP ----
load_dotenv()
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Agentic PDF + Web Chatbot", layout="wide")
st.title("📄 PDF Analysis & 🌐 Web Search Assistant")

# ---- SESSION STATE ----
for key, default in {
    "store": {},
    "vectors_store": None,
    "messages": [{"role": "assistant", "content": "I can analyze PDFs and search the web."}]
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---- HELPERS ----
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={"batch_size": 64, "normalize_embeddings": True},
    )


def fingerprint(files):
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.getvalue())
    return h.hexdigest()


@st.cache_resource
def build_vectorstore(fp, file_bytes):
    embeddings = get_embeddings()
    docs = []

    for _, b in file_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(b)
            path = tmp.name
        docs.extend(PyPDFLoader(path).load())
        os.unlink(path)

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120
    ).split_documents(docs)

    return FAISS.from_documents(chunks, embeddings)


def history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]


# ---- TOOLS ----
search = DuckDuckGoSearchRun()
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1))
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1))

# ---- SIDEBAR ----
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))

    # ✅ Use EXACT models you provided
    MODELS = {
        "llama-3.1-8b-instant": "llama-3.1-8b-instant",
        "openai/gpt-oss-120b": "openai/gpt-oss-120b",
        "moonshotai/kimi-k2-instruct-0905": "moonshotai/kimi-k2-instruct-0905",
        "meta-llama/llama-4-maverick-17b-128e-instruct": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct": "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-prompt-guard-2-86m": "meta-llama/llama-prompt-guard-2-86m",
        "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
    }

    model_name = st.selectbox("Model", list(MODELS.keys()), index=0)
    model = MODELS[model_name]

    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    use_pdf = st.toggle("Prefer PDF answers", True)


# ---- PDF UPLOAD ----
with st.form("pdf_form"):
    pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    submit = st.form_submit_button("Process PDFs")

if submit and pdfs:
    fp = fingerprint(pdfs)
    files = [(f.name, f.getvalue()) for f in pdfs]
    with st.spinner("Processing PDFs..."):
        st.session_state.vectors_store = build_vectorstore(fp, files)
    st.success("PDFs ready!")


# ---- CHAT HISTORY ----
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])


# ---- CHAT ----
if not api_key:
    st.warning("Enter your Groq API key.")
    st.stop()

llm = ChatGroq(
    groq_api_key=api_key,
    model=model,
    temperature=temperature,
    streaming=True,
)

agent = initialize_agent(
    tools=[search, arxiv, wiki],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handling_parsing_errors=True,
)

rag_chain = None
if st.session_state.vectors_store:
    retriever = st.session_state.vectors_store.as_retriever(k=3)

    # ✅ FIX: separate prompts (context prompt MUST include {context}; retriever prompt MUST NOT)
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, rewrite it as a standalone question. "
         "Do NOT answer the question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer using ONLY the provided PDF context. "
         "If the answer is not in the context, say: 'Not found in the uploaded PDFs.'\n\n"
         "{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    doc_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, doc_chain)

    rag_chain = RunnableWithMessageHistory(
        rag_chain,
        history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        cb = StreamlitCallbackHandler(st.container())
        try:
            if use_pdf and rag_chain:
                out = rag_chain.invoke({"input": prompt}, {"configurable": {"session_id": "default"}})
                answer = out.get("answer", "")
                if not answer:
                    answer = "Not found in the uploaded PDFs."
            else:
                answer = agent.run(prompt, callbacks=[cb])
        except Exception as e:
            if "decommissioned" in str(e) or "model_decommissioned" in str(e):
                st.warning("Model retired. Switching to llama-3.1-8b-instant.")
                llm = ChatGroq(
                    groq_api_key=api_key,
                    model="llama-3.1-8b-instant",
                    temperature=temperature,
                    streaming=True,
                )
                agent = initialize_agent(
                    tools=[search, arxiv, wiki],
                    llm=llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    handling_parsing_errors=True,
                )
                if use_pdf and st.session_state.vectors_store:
                    retriever = st.session_state.vectors_store.as_retriever(k=3)
                    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
                    doc_chain = create_stuff_documents_chain(llm, qa_prompt)
                    rag_chain_inner = create_retrieval_chain(history_aware_retriever, doc_chain)
                    rag_chain = RunnableWithMessageHistory(
                        rag_chain_inner,
                        history,
                        input_messages_key="input",
                        history_messages_key="chat_history",
                        output_messages_key="answer",
                    )
                answer = agent.run(prompt, callbacks=[cb])
            else:
                answer = f"Error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.write(answer)
