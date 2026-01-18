import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Import ConversationBufferMemory from the local memory.py module
from .memory import ConversationBufferMemory  # Relative import


class ConversationalRetrievalChain:
    """
    ConversationalRetrievalChain implementation that combines:
    - Document retrieval from vector store
    - Language model for generation
    - Conversation memory for context awareness
    """
    def __init__(self, llm, retriever, memory, return_source_documents=True):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.return_source_documents = return_source_documents

        # Create conversational prompt template
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. Use the following context from retrieved documents to answer questions accurately.

Context from documents:
{context}

Answer based on the context above. If the answer is not in the context, say so."""),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ])

        # Build the RAG chain
        self.chain = (
            RunnablePassthrough.assign(
                context=lambda x: self._format_docs(self.retriever.invoke(x["question"]))
            )
            | self.qa_prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs):
        """Format retrieved documents into context string"""
        return "\n\n".join(doc.page_content for doc in docs)

    @classmethod
    def from_llm(cls, llm, retriever, memory, return_source_documents=True, **kwargs):
        """
        Factory method to create ConversationalRetrievalChain.
        Matches the original LangChain API.
        """
        return cls(llm, retriever, memory, return_source_documents)

    def invoke(self, inputs):
        """
        Run the conversational RAG chain with memory.

        Args:
            inputs: Dict with 'question' or 'query' key

        Returns:
            Dict with 'answer', 'result', and optionally 'source_documents'
        """
        question = inputs.get("question", inputs.get("query", ""))

        # Load conversation history from memory
        history = self.memory.load_memory_variables({})
        chat_history = history.get(self.memory.memory_key, [])

        # Retrieve relevant documents
        docs = self.retriever.invoke(question)

        # Run the conversational chain
        answer = self.chain.invoke({
            "question": question,
            "chat_history": chat_history
        })

        # Save this turn to memory
        self.memory.save_context(
            {"question": question},
            {self.memory.output_key: answer}
        )

        # Return result in expected format
        result = {
            "answer": answer,
            "result": answer,  # Some code expects 'result' key
            "question": question
        }

        if self.return_source_documents:
            result["source_documents"] = docs

        return result

    def __call__(self, inputs):
        """Allow chain to be called directly"""
        return self.invoke(inputs)

    def __repr__(self):
        return f"ConversationalRetrievalChain(llm={self.llm.model_name}, memory={self.memory})"
