import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.schema import Document

# Available LLM models
AVAILABLE_MODELS = {
    'gemma2-9b-it': {
        'name': 'gemma2-9b-it',
        'max_tokens': 1024,
        'default_temperature': 0.3
    },
    'llama-3.1-8b-instant': {
        'name': 'llama-3.1-8b-instant',
        'max_tokens': 1000,
        'default_temperature': 0.3
    },
    'deepseek-r1-distill-qwen-32b': {
        'name': 'deepseek-r1-distill-qwen-32b',
        'max_tokens': 2048,
        'default_temperature': 0.3
    },
    'llama-3.3-70b-versatile': {
        'name': 'llama-3.3-70b-versatile',
        'max_tokens': 2048,
        'default_temperature': 0.3
    }
}

class LLMManager:
    def __init__(self, db_path):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.db_path = db_path
        self.chat_history = []
        
        # Initialize with default model
        default_model = AVAILABLE_MODELS['llama-3.1-8b-instant']
        self.current_model = default_model['name']
        self.current_temperature = default_model['default_temperature']
        
        # Initialize LLM
        self.llm = ChatGroq(
            temperature=self.current_temperature,
            groq_api_key=self.api_key,
            model_name=self.current_model,
            max_tokens=default_model['max_tokens']
        )
        
        # Initialize embedding and ChromaDB
        self.embedding = FastEmbedEmbeddings()
        self.db = Chroma(
            embedding_function=self.embedding,
            persist_directory=self.db_path
        )
        
        # Initialize prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are conversing with an AI assistant who knows Sanskrit and Ayurveda. The assistant is here to help with queries about Sanskrit texts and Ayurvedic knowledge."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}\nContext: {context}")
        ])
        
        # Ensure DB is populated
        self.ensure_db_populated()
    
    def ensure_db_populated(self):
        """Ensures ChromaDB is preloaded with initial data."""
        if not self.db.get()["documents"]:
            self.db.add_documents([
                Document(
                    page_content="संस्कृत ज्ञानकोशः आरम्भः। आयुर्वेद ज्ञानम्।",
                    metadata={"source": "default"}
                )
            ])
            self.db.persist()
    
    def update_settings(self, model=None, temperature=None):
        """
        Updates LLM settings.
        
        Args:
            model: The model name to use
            temperature: The temperature setting for the model
        
        Returns:
            dict: Updated settings
        """
        # Use existing values if new ones aren't provided
        model_name = model if model is not None else self.current_model
        temp_value = temperature if temperature is not None else self.current_temperature
        
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Invalid model name: {model_name}")
        
        model_config = AVAILABLE_MODELS[model_name]
        self.current_model = model_name
        self.current_temperature = temp_value
        
        self.llm = ChatGroq(
            temperature=temp_value,
            groq_api_key=self.api_key,
            model_name=model_name,
            max_tokens=model_config['max_tokens']
        )
        
        return {
            "model": model_name,
            "temperature": temp_value,
            "max_tokens": model_config['max_tokens']
        }
    
    def process_query(self, query):
        """Processes a query and returns a response."""
        # Create retriever
        retriever = self.db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.1}
        )
        
        # Create history-aware retriever
        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", "Generate a search query to find relevant context information.")
        ])
        
        history_retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=retriever,
            prompt=retriever_prompt
        )
        
        # Process query
        input_data = {
            "input": query,
            "context": "",
            "chat_history": self.chat_history
        }
        
        retrieval_chain = create_retrieval_chain(
            history_retriever,
            create_stuff_documents_chain(self.llm, self.prompt_template)
        )
        
        result = retrieval_chain.invoke(input_data)
        
        # Update chat history
        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=result["answer"]))
        
        return result["answer"]
    
    def clear_history(self):
        """Clears the chat history."""
        self.chat_history = []
        return True
    
    def add_documents(self, documents):
        """Adds new documents to the vector store."""
        self.db.add_documents(documents)
        self.db.persist()
        return True
    
    def add_document_to_context(self, text):
        """
        Adds a document text to the context.
        
        Args:
            text: The text content to add
            
        Returns:
            bool: True if successful
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        
        # Create document chunks
        chunks = text_splitter.create_documents([text])
        
        # Add to vector store
        return self.add_documents(chunks) 