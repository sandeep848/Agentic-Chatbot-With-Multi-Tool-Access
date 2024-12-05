import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
import tempfile
from dotenv import load_dotenv
################################################################################################

if 'store' not in st.session_state:
    st.session_state.store = {}
if 'vectors_store' not in st.session_state:
    st.session_state.vectors_store = None
if "messages" not in st.session_state:
    st.session_state.messages = [
        {'role': "assistant", "content": "I'm a chatbot who can search the web and analyze PDFs"}
    ]

st.set_page_config(page_title="Smart Chat Assistant", layout="wide")
st.title("PDF Analysis & Web Search Assistant")

with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Enter your Groq API key:", type="password")
    engine = st.selectbox(
        "Select model",
        ['llama-3.1-70b-versatile', 'llama-3.1-8b-instant', 'gemma-7b-it', 
         'llama-3.2-90b-vision-preview', 'llama3-70b-8192', "mixtral-8x7b-32768"]
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
    st.divider()
    session_id = st.text_input("Session ID", value="default_session",
                              help="Use this to maintain separate chat histories")


arxiv_api = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_api)
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=wiki_api)
search = DuckDuckGoSearchRun(name="Search")

try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Error initializing embeddings: {str(e)}")
    st.stop()

def process_pdfs(uploaded_files):
    """Process uploaded PDF files and create vector store"""
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            documents.extend(docs)
        finally:
            os.unlink(tmp_path)
    
    if not documents:
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(documents)
    return FAISS.from_documents(splits, embeddings)

def get_session_history(session: str) -> BaseChatMessageHistory:
    """Get or create chat history for session"""
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

st.subheader("Upload PDFs for Analysis")
uploaded_files = st.file_uploader(
    "Choose PDF files",
    type="pdf",
    accept_multiple_files=True,
    help="Upload one or more PDF files to chat with"
)

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        vector_store = process_pdfs(uploaded_files)
        if vector_store:
            st.session_state.vectors_store = vector_store
            st.success("PDFs processed successfully!")
        else:
            st.error("Failed to process PDFs")

if api_key:
    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model=engine,
            temperature=temperature,
            streaming=True
        )

       
        tools = [search, arxiv, wiki]
        search_agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handling_parsing_errors=True
        )

        if st.session_state.vectors_store:
            retriever = st.session_state.vectors_store.as_retriever(
                search_kwargs={"k": 3}
            )

            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", "Given a chat history and the latest user question, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.\n\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt
            )
            
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(
                history_aware_retriever, question_answer_chain
            )

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg['content'])

        if prompt := st.chat_input("Ask me anything..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                
                if st.session_state.vectors_store and "pdf" in prompt.lower():
                    try:
                        response = conversational_rag_chain.invoke(
                            {"input": prompt},
                            config={"configurable": {"session_id": session_id}}
                        )
                        final_response = response['answer']
                    except Exception as e:
                        st.error(f"Error in PDF processing: {str(e)}")
                        final_response = "I encountered an error processing the PDFs. Let me try searching the web instead."
                        response = search_agent.run(prompt, callbacks=[st_cb])
                        final_response += "\n\n" + response
                else:
                    response = search_agent.run(prompt, callbacks=[st_cb])
                    final_response = response

                st.session_state.messages.append({"role": "assistant", "content": final_response})
                st.write(final_response)

    except Exception as e:
        st.error(f"Error in chat interface: {str(e)}")

elif not api_key:
    st.warning("Please enter your Groq API key in the sidebar.")
    ##
