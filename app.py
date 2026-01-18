import sys
import os
import streamlit as st
import hashlib
import time
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import modules from src
from src.memory import ConversationBufferMemory
from src.rag_chain import ConversationalRetrievalChain
from src.utils import load_and_split_documents, get_vectorstore
from src.llm_setup import get_openai_api_key, initialize_embeddings, initialize_llm


# ============================================================================
# SECURITY: PASSWORD PROTECTION
# ============================================================================

def check_password():
    """Returns True if the user has the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        entered_hash = hashlib.sha256(st.session_state["password"].encode()).hexdigest()
        
        # IMPORTANT: Replace this hash with your own password hash
        # To generate: import hashlib; print(hashlib.sha256(b"your_password").hexdigest())
        correct_hash = "92276f47fd108e1e4e5b1e62404fecfd887d4e67599622d466bd853dde978a0a"  # Default: "password"
        
        if entered_hash == correct_hash:
            st.session_state["password_correct"] = True
            st.session_state["login_time"] = datetime.now()
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    # Check if password is correct
    if "password_correct" not in st.session_state:
        st.markdown("# 🔐 RAG Chatbot Access")
        st.markdown("Please enter the password to continue")
        st.text_input(
            "Password", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.info("💡 Default password is: **password** (change this in production!)")
        return False
    elif not st.session_state["password_correct"]:
        st.markdown("# 🔐 RAG Chatbot Access")
        st.text_input(
            "Password", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("❌ Incorrect password. Please try again.")
        return False
    else:
        return True


# ============================================================================
# SECURITY: RATE LIMITING
# ============================================================================

def check_rate_limit():
    """Enforce rate limiting: minimum time between queries."""
    
    RATE_LIMIT_SECONDS = 3  # Minimum seconds between queries
    
    if 'last_query_time' not in st.session_state:
        st.session_state.last_query_time = 0
    
    time_since_last = time.time() - st.session_state.last_query_time
    
    if time_since_last < RATE_LIMIT_SECONDS:
        wait_time = RATE_LIMIT_SECONDS - int(time_since_last)
        return False, wait_time
    
    return True, 0


def update_query_time():
    """Update the last query timestamp."""
    st.session_state.last_query_time = time.time()


# ============================================================================
# SECURITY: USAGE TRACKING
# ============================================================================

def initialize_usage_tracking():
    """Initialize usage tracking in session state."""
    if 'query_count' not in st.session_state:
        st.session_state.query_count = 0
    if 'session_start' not in st.session_state:
        st.session_state.session_start = datetime.now()


def increment_query_count():
    """Increment the query counter."""
    st.session_state.query_count += 1


def display_usage_stats():
    """Display usage statistics in the sidebar."""
    if 'query_count' in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Usage Statistics")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Queries", st.session_state.query_count)
        
        if 'session_start' in st.session_state:
            duration = datetime.now() - st.session_state.session_start
            minutes = int(duration.total_seconds() / 60)
            with col2:
                st.metric("Session", f"{minutes}m")
        
        # Warning for high usage
        if st.session_state.query_count > 50:
            st.sidebar.warning("⚠️ High usage detected. Consider monitoring API costs.")


# ============================================================================
# PASSWORD CHECK - RUNS FIRST
# ============================================================================

if not check_password():
    st.stop()  # Stop execution if password is incorrect


# ============================================================================
# MAIN APP (Only runs after successful authentication)
# ============================================================================

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG-Powered Chatbot")
st.markdown("Ask questions about your documents!")

# Initialize usage tracking
initialize_usage_tracking()

# Initialize session state
if 'conversational_chain' not in st.session_state:
    st.session_state.conversational_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Sidebar for initialization and controls
with st.sidebar:
    st.header("⚙️ Setup")
    
    # Display logged-in status
    if 'login_time' in st.session_state:
        st.success("✓ Authenticated")
    
    if st.button("Initialize RAG System"):
        with st.spinner("Initializing RAG components..."):
            try:
                # Get API key
                api_key = get_openai_api_key()
                if not api_key:
                    st.error("OpenAI API key not found. Please set it in environment variables or Colab userdata.")
                    st.stop()
                
                # Initialize embeddings and LLM
                embeddings = initialize_embeddings(api_key=api_key)
                llm = initialize_llm(api_key=api_key)
                st.success("✓ LLM and embeddings initialized")
                
                # Use local Colab storage path
                rag_docs_path = "/RAG_Docs/"
                
                # Check if documents exist
                if not os.path.exists(rag_docs_path):
                    st.error(f"Document path does not exist: {rag_docs_path}")
                    st.info("Please copy RAG_Docs to /RAG_Docs/ first")
                    st.stop()
                
                # Load and split documents
                chunks = load_and_split_documents(rag_docs_path)
                if not chunks:
                    st.error("No documents loaded")
                    st.stop()
                st.success(f"✓ Loaded {len(chunks)} document chunks")
                
                # Create vector store
                vectorstore = get_vectorstore(chunks, embeddings)
                st.success("✓ Vector store created")
                
                # Initialize memory and chain
                memory = ConversationBufferMemory()
                conversational_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever(),
                    memory=memory
                )
                
                st.session_state.conversational_chain = conversational_chain
                st.session_state.initialized = True
                st.success("✓ RAG system ready!")
                
            except Exception as e:
                st.error(f"Error during initialization: {str(e)}")
                st.exception(e)
    
    if st.session_state.initialized:
        st.success("System is ready!")
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            if st.session_state.conversational_chain:
                st.session_state.conversational_chain.memory.clear()
            st.rerun()
    
    # Display usage statistics
    display_usage_stats()

# Main chat interface
if st.session_state.initialized:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Check rate limit
        can_proceed, wait_time = check_rate_limit()
        
        if not can_proceed:
            st.warning(f"⏳ Please wait {wait_time} seconds before sending another query.")
            st.stop()
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Update rate limit timer
        update_query_time()
        
        # Increment usage counter
        increment_query_count()
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversational_chain.invoke({"question": prompt})
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    # Show source documents in expander
                    if "source_documents" in response:
                        with st.expander("📚 Source Documents"):
                            for i, doc in enumerate(response["source_documents"]):
                                st.markdown(f"**Document {i+1}:**")
                                st.text(doc.page_content[:300] + "...")
                                st.markdown("---")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    answer = "Sorry, I encountered an error processing your question."
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Please initialize the RAG system using the sidebar button")
    
    # Show helpful information
    with st.expander("ℹ️ How to use this chatbot"):
        st.markdown("""
        1. **Initialize the system**: Click the "Initialize RAG System" button in the sidebar
        2. **Ask questions**: Use the chat input at the bottom to ask questions about your documents
        3. **View sources**: Expand "Source Documents" to see which documents were used
        4. **Clear history**: Use the "Clear Chat History" button to start fresh
        
        **Security Features:**
        - 🔐 Password protection
        - ⏱️ Rate limiting (3 seconds between queries)
        - 📊 Usage tracking
        """)
