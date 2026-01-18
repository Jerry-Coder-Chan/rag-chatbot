from langchain_core.messages import HumanMessage, AIMessage


class ConversationBufferMemory:
    """
    ConversationBufferMemory implementation for maintaining conversation history.
    Stores chat history and provides it to the conversational chain.
    """
    def __init__(self, memory_key="chat_history", return_messages=True, output_key="answer"):
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.output_key = output_key
        self.chat_history = []

    def save_context(self, inputs, outputs):
        """Save a conversation turn to memory"""
        question = inputs.get("question", inputs.get("query", ""))
        answer = outputs.get(self.output_key, outputs.get("answer", outputs.get("result", "")))
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer})

    def load_memory_variables(self, inputs=None):
        """Load conversation history as messages"""
        messages = []
        for msg in self.chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        return {self.memory_key: messages}

    def clear(self):
        """Clear all conversation history"""
        self.chat_history = []

    def __repr__(self):
        return f"ConversationBufferMemory(messages={len(self.chat_history)})"
