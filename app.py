import streamlit as st
import os
import tempfile
import csv
import json
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.schema import Document

load_dotenv()
openai_api_key = os.environ["OPENAI_API_KEY"]

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    .sidebar-header {
        color: #2E86AB;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .chat-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .upload-section {
        background: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #2E86AB;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stTextInput > div > div > input {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2E86AB;
        box-shadow: 0 0 10px rgba(46, 134, 171, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #2E86AB, #A23B72);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .file-upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .status-indicator {
        background: #28a745;
        color: white;
        border-radius: 20px;
        padding: 5px 15px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 5px 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        color: #856404;
    }
    
    /* Fix streamlit-chat message visibility */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #333 !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 10px !important;
        margin: 10px 0 !important;
    }
    
    /* Style for streamlit_chat library messages */
    div[data-testid="stChatMessageContent"] {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #333 !important;
        border-radius: 10px !important;
        padding: 15px !important;
        margin: 8px 0 !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    div[data-testid="stChatMessageContent"] p {
        color: #333 !important;
        margin: 0 !important;
    }
    
    /* Avatar styling */
    .stChatMessage [data-testid="chatAvatarIcon-user"] {
        background-color: #2E86AB !important;
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        background-color: #A23B72 !important;
    }
    
    /* Fix input field styling */
    .stTextInput input {
        background-color: #ffffff !important;
        color: #333 !important;
        border: 2px solid #e9ecef !important;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []

    if "generated" not in st.session_state:
        st.session_state["generated"] = [
            "Hello! Feel free to ask me any questions."
        ]

    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey! 👋"]


def conversation_chat(query, chain, history):
    result = chain({
        "question": query,
        "chat_history": history
    })
    history.append((query, result["answer"]))
    return result["answer"]


def display_chat_history(chain):
    # Chat interface with enhanced styling
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    reply_container = st.container()
    container = st.container()

    with container:
        st.markdown("### 💬 Chat with your Documents")
        
        with st.form(key="my_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_input(
                    "",
                    placeholder="💭 Ask me anything about your uploaded documents...",
                    key="input",
                    label_visibility="collapsed"
                )
            
            with col2:
                submit_button = st.form_submit_button(
                    label="🚀 Send",
                    use_container_width=True
                )

        if submit_button and user_input:
            with st.spinner("🤔 Thinking... Generating response"):
                output = conversation_chat(
                    query=user_input,
                    chain=chain,
                    history=st.session_state["history"]
                )

            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output)

    if st.session_state["generated"]:
        with reply_container:
            st.markdown("### 📝 Conversation History")
            
            for i in range(len(st.session_state["generated"])):
                # User message
                message(
                    st.session_state["past"][i],
                    is_user=True,
                    key=str(i) + "_user",
                    avatar_style="personas"
                )
                # Bot response
                message(
                    st.session_state["generated"][i],
                    key=str(i),
                    avatar_style="bottts"
                )
    
    st.markdown('</div>', unsafe_allow_html=True)


def create_conversational_chain(vector_store):
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.1,
        openai_api_key=openai_api_key
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )

    return chain


def main():
    # Page configuration
    st.set_page_config(
        page_title="🤖 RAG ChatBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    # Main header with enhanced styling
    st.markdown('<h1 class="main-header">🤖 RAG ChatBot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">🚀 Powered by LangChain & ChatGPT | Upload documents and ask questions!</p>', unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Enhanced sidebar
        st.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: white; margin-top: 0;">📁 Document Processing</h3>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload Your Documents",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx', 'doc', 'csv', 'json'],
            help="Supported formats: PDF, TXT, DOCX, DOC, CSV, JSON"
        )
        
        if uploaded_files:
            st.markdown('<div class="status-indicator">✅ Files Uploaded Successfully!</div>', unsafe_allow_html=True)
            st.markdown(f"**📊 Files Count:** {len(uploaded_files)}")
            
            # Show uploaded file names
            for file in uploaded_files:
                st.markdown(f"• 📄 {file.name}")
        else:
            st.markdown('<div class="warning-box">⚠️ Please upload documents to start chatting!</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Processing info
        if uploaded_files:
            with st.expander("🔧 Processing Details", expanded=False):
                st.markdown("""
                **Document Processing Steps:**
                1. 📥 File Upload & Extraction
                2. 🔪 Text Chunking (768 chars)
                3. 🧠 Embedding Generation
                4. 💾 Vector Store Creation
                5. 🤖 ChatGPT Integration
                """)
    
    with col2:
        if uploaded_files:
            # Process documents
            with st.spinner("🔄 Processing your documents... This may take a moment"):
                text = []
                progress_bar = st.progress(0)
                
                for idx, file in enumerate(uploaded_files):
                    file_extension = os.path.splitext(file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(file.read())
                        temp_file_path = temp_file.name

                    loader = None
                    if file_extension == ".pdf":
                        loader = PyPDFLoader(temp_file_path)
                    elif file_extension == ".docx" or file_extension == ".doc":
                        loader = Docx2txtLoader(temp_file_path)
                    elif file_extension == ".txt":
                        loader = TextLoader(temp_file_path)

                    # CSV / JSON support: parse into text and wrap into langchain Document
                    if file_extension == ".csv":
                        try:
                            with open(temp_file_path, newline='', encoding='utf-8') as f:
                                reader = csv.reader(f)
                                rows = [', '.join(row) for row in reader]
                                content = "\n".join(rows)
                            text.append(Document(page_content=content))
                        except Exception:
                            # fallback: read raw
                            with open(temp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            text.append(Document(page_content=content))
                        finally:
                            os.remove(temp_file_path)

                    elif file_extension == ".json":
                        try:
                            with open(temp_file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                content = json.dumps(data, indent=2, ensure_ascii=False)
                            text.append(Document(page_content=content))
                        except Exception:
                            with open(temp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            text.append(Document(page_content=content))
                        finally:
                            os.remove(temp_file_path)

                    else:
                        if loader:
                            text.extend(loader.load())
                        # remove temp file for other loaders
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / len(uploaded_files))

                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=768,
                    chunk_overlap=128,
                    length_function=len
                )
                text_chunks = text_splitter.split_documents(text)

                embedding = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"}
                )

                vector_store = Chroma.from_documents(
                    documents=text_chunks,
                    embedding=embedding,
                    persist_directory="chroma_store"
                )

                chain = create_conversational_chain(vector_store=vector_store)
                
                # Clear progress bar
                progress_bar.empty()
                
                # Success message
                st.success(f"✅ Successfully processed {len(text_chunks)} text chunks from {len(uploaded_files)} documents!")

            # Display chat interface
            display_chat_history(chain=chain)
        else:
            # Welcome message when no files are uploaded
            st.markdown("""
            <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px; margin: 20px 0;">
                <h2 style="color: #2E86AB;">👋 Welcome to RAG ChatBot!</h2>
                <p style="font-size: 1.1rem; color: #666; margin: 20px 0;">
                    🚀 Get started by uploading your documents on the left sidebar.<br>
                    📚 Supported formats: PDF, TXT, DOCX, DOC<br>
                    💬 Once uploaded, you can ask questions about your documents!
                </p>
                <div style="background: #e3f2fd; padding: 20px; border-radius: 10px; margin: 20px 0;">
                    <h4 style="color: #1976d2; margin: 0;">✨ Features:</h4>
                    <ul style="text-align: left; color: #555; margin: 10px 0;">
                        <li>🤖 AI-powered document analysis</li>
                        <li>💭 Natural language queries</li>
                        <li>📊 Multiple document support</li>
                        <li>🧠 Contextual conversations</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
