import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def get_openai_api_key():
    """Retrieves the OpenAI API key from multiple sources."""
    # Try environment variable first (works in all environments)
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # Try Colab userdata (only works in Colab)
    try:
        from google.colab import userdata
        api_key = userdata.get('OPENAI_API_KEY')
        if api_key:
            return api_key
    except ImportError:
        # Not in Colab environment, that's fine
        pass
    except Exception as e:
        # userdata.get() failed, key might not exist
        print(f"Warning: Could not retrieve from Colab userdata: {e}")
    
    # Try Streamlit secrets (if running in Streamlit)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            return st.secrets['OPENAI_API_KEY']
    except ImportError:
        # Not in Streamlit environment
        pass
    except Exception:
        # Streamlit secrets not configured
        pass
    
    return None


def initialize_embeddings(api_key=None, model="text-embedding-ada-002"):
    """Initializes and returns an OpenAIEmbeddings instance."""
    if api_key is None:
        api_key = get_openai_api_key()
    if not api_key:
        raise ValueError("OpenAI API key not provided or found.")
    return OpenAIEmbeddings(api_key=api_key, model=model)


def initialize_llm(api_key=None, model="gpt-4-turbo", temperature=0.7):
    """Initializes and returns a ChatOpenAI (LLM) instance."""
    if api_key is None:
        api_key = get_openai_api_key()
    if not api_key:
        raise ValueError("OpenAI API key not provided or found.")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=temperature
    )
