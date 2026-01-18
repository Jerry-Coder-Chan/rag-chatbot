import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def clean_special_tokens(text):
    """Remove OpenAI special tokens from text"""
    special_tokens = ['<|endoftext|>', '<|fim_prefix|>', '<|fim_middle|>', '<|fim_suffix|>', '<|endofprompt|>']
    for token in special_tokens:
        text = text.replace(token, '')
    return text


def load_and_split_documents(rag_docs_path, chunk_size=500, chunk_overlap=100):
    """
    Loads PDF documents from a specified path, cleans them, and splits them into chunks.
    """
    all_documents = []
    if not os.path.exists(rag_docs_path):
        print(f"Warning: Document path '{rag_docs_path}' does not exist.")
        return []

    for filename in os.listdir(rag_docs_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(rag_docs_path, filename)
            if os.path.exists(file_path):
                print(f"Loading document: {filename}")
                loader = PyMuPDFLoader(file_path)
                all_documents.extend(loader.load())
            else:
                print(f"Warning: File not found at {file_path}. Skipping.")

    if not all_documents:
        print(f"Error: No PDF documents found in '{rag_docs_path}' or no documents could be loaded.")
        return []

    # Clean documents before splitting
    # The cleaning is applied before splitting to ensure token boundaries are consistent
    for doc in all_documents:
        doc.page_content = clean_special_tokens(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"Number of chunks created: {len(chunks)}")
    return chunks


def get_vectorstore(chunks, embeddings_model, faiss_index_path="faiss_index"):
    """
    Creates or loads a FAISS vector store from document chunks.
    If a local FAISS index exists, it loads it. Otherwise, it creates a new one
    from the provided chunks and saves it.
    """
    # Ensure embeddings_model is passed correctly. For loading, an instance is needed.
    # In a real app, you'd ensure embeddings_model is available before calling this.

    if os.path.exists(faiss_index_path) and os.path.exists(os.path.join(faiss_index_path, "index.faiss")):
        print(f"Loading existing FAISS index from {faiss_index_path}...")
        # Need an embeddings model instance to load the index
        vectorstore = FAISS.load_local(faiss_index_path, embeddings_model, allow_dangerous_deserialization=True)
        print("✓ FAISS index loaded.")
    else:
        print("Creating new FAISS index...")
        if not chunks:
            raise ValueError("No chunks provided to create a new vector store.")
        vectorstore = FAISS.from_documents(chunks, embeddings_model)
        vectorstore.save_local(faiss_index_path)
        print(f"✓ New FAISS index created and saved to {faiss_index_path} with {len(chunks)} chunks")
    return vectorstore
