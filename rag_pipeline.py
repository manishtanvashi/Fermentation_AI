# ─────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────
import os
import uuid
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────
# LOAD ENVIRONMENT VARIABLES
# ─────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────
# STEP 1 — LOAD DOCUMENTS
# ─────────────────────────────────────────
def load_documents(pdf_directory: str = "./Data/PDFs"):
    # Configure loader to find all PDFs in directory
    pdf_loader = DirectoryLoader(
        pdf_directory,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=False
    )
    # Actually load all PDFs → returns list of Document objects
    documents = pdf_loader.load()
    print(f"Loaded {len(documents)} pages from PDFs")
    return documents

# ─────────────────────────────────────────
# STEP 2 — CHUNK DOCUMENTS
# ─────────────────────────────────────────
def split_documents(documents):
    # Configure splitter → chunk_size=1000 chars, chunk_overlap=200 chars
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    # Actually split all documents into chunks
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks

# ─────────────────────────────────────────
# STEP 3 — EMBEDDING MANAGER CLASS
# ─────────────────────────────────────────
class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Save model name for later use
        self.model_name = model_name
        # Empty slot — model not loaded yet
        self.model = None
        # Load model automatically when object created
        self._load_model()

    def _load_model(self):
        # Private method — only called inside class
        try:
            print(f"Loading Embedding model: {self.model_name}")
            # Download and load model from HuggingFace
            self.model = SentenceTransformer(self.model_name)
            print(f"Model Loaded Successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            # If loading fails → show error and stop
            print(f"Error Loading Model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        # Safety check — can't embed without model
        if not self.model:
            raise ValueError("Model Not Loaded")
        # Convert list of texts → numpy array of vectors (shape: len(texts), 384)
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings

# ─────────────────────────────────────────
# STEP 4 — VECTOR STORE CLASS
# ─────────────────────────────────────────
class VectorStore:
    """Manages ChromaDB vector store for storing and searching embeddings"""

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = "./data/vector_store"
    ):
        # Save collection name and folder path
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        # Empty slots — not connected yet
        self.client = None
        self.collection = None
        # Set up ChromaDB automatically when object created
        self._initialize_store()

    def _initialize_store(self):
        # Private method — only called inside class
        try:
            # Create folder if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            # Connect ChromaDB to persist directory → saves vectors to disk
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            # Get existing collection OR create new one
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        # Skip if documents already loaded → saves time on restart
        if self.collection.count() > 0:
            print("Documents already exist — skipping!")
            return

        print(f"Adding {len(documents)} documents to vector store...")

        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID for each chunk
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            # Save metadata with extra info
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)
            # Save chunk text
            documents_text.append(doc.page_content)
            # Save vector
            embeddings_list.append(embedding.tolist())

        try:
            # Store everything in ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise

# ─────────────────────────────────────────
# STEP 5 — RAG RETRIEVER CLASS
# ─────────────────────────────────────────
class RAGRetriever:
    """Handles query based retrieval from vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        # Reuse existing vector store (with 365 chunks already loaded)
        self.vector_store = vector_store
        # Reuse existing embedding manager (model already loaded)
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        print(f"Retrieving documents for query: '{query}'")

        # Convert user question to vector (384 dimensions)
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            # Search ChromaDB → find top_k most similar chunks
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            # Extract results from ChromaDB response
            retrieved_docs = []
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            ids = results.get("ids", [[]])[0]

            # Process each retrieved chunk
            for i, (doc_id, document, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances)
            ):
                # Convert distance to similarity (1 = perfect, 0 = no match)
                similarity_score = 1 - distance

                retrieved_docs.append({
                    "id": doc_id,
                    "content": document,
                    "metadata": metadata,
                    "similarity_score": similarity_score,
                    "distance": distance,
                    "rank": i + 1
                })

            print(f"Retrieved {len(retrieved_docs)} documents")
            return retrieved_docs

        except Exception as e:
            print(f"Error during Retrieval: {e}")
            return []

    def get_relevant_documents(self, query: str):
        # LangChain compatible method → returns Document objects
        results = self.retrieve(query)
        if not results:
            return []
        # Convert dict results → LangChain Document objects
        return [
            Document(
                page_content=r["content"],
                metadata=r["metadata"]
            )
            for r in results
        ]

# ─────────────────────────────────────────
# STEP 6 — PROMPT TEMPLATE
# ─────────────────────────────────────────

# Tell LLM how to behave and what to do with context
RAG_PROMPT = """You are a helpful assistant specializing in fermentation research.
Use the provided context to answer the user's question accurately.
If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question:
{query}

Answer:"""

# ─────────────────────────────────────────
# STEP 7 — GET ANSWER FUNCTION
# ─────────────────────────────────────────
def get_answer(query: str, retriever: RAGRetriever, llm: ChatGroq) -> Dict[str, Any]:

    # Retrieve relevant chunks from ChromaDB
    docs = retriever.get_relevant_documents(query)

    # If nothing found → return friendly message
    if not docs:
        return {
            "answer": "No relevant information found.",
            "sources": []
        }

    # Join all chunks into one big context text
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # Get unique source files
    sources = list(set([
        doc.metadata.get("source", "Unknown")
        for doc in docs
    ]))

    # Build LangChain pipeline → prompt | llm | parser
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    chain = prompt | llm | StrOutputParser()

    # Run chain → fill prompt → send to LLM → get string answer
    answer = chain.invoke({
        "context": context_text,
        "query": query
    })

    # Return answer and sources
    return {
        "answer": answer,
        "sources": sources
    }

# ─────────────────────────────────────────
# STEP 8 — INITIALIZE EVERYTHING
# ─────────────────────────────────────────
def initialize_rag():
    """Initialize complete RAG pipeline — call this once at startup"""

    # Initialize embedding manager → loads SentenceTransformer model
    embedding_manager = EmbeddingManager()

    # Initialize vector store → connects to ChromaDB
    vector_store = VectorStore()

    # Only load PDFs if ChromaDB is empty
    if vector_store.collection.count() == 0:
        print("No documents found — loading PDFs...")
        documents = load_documents()
        chunks = split_documents(documents)
        # Generate embeddings for all chunks
        embeddings = embedding_manager.generate_embeddings(
            [chunk.page_content for chunk in chunks]
        )
        # Store in ChromaDB
        vector_store.add_documents(chunks, embeddings)

    # Initialize retriever → connects vector store + embedding manager
    retriever = RAGRetriever(vector_store, embedding_manager)

    # Initialize LLM → connects to Groq API
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=1024,
        api_key=os.getenv("GROQ_API_KEY")
    )

    print("RAG pipeline ready!")
    # Return both for use in FastAPI and Streamlit
    return retriever, llm

# ─────────────────────────────────────────
# TESTING — Remove this after testing!
# ─────────────────────────────────────────
#if __name__ == "__main__":
#    print("Testing RAG pipeline...")
    
    # Initialize everything
 #   retriever, llm = initialize_rag()
    
    # Test a question
  #  result = get_answer(
   #     "What is fermentation?",
    #    retriever,
     #   llm
    #)
    
    #print("\n─── ANSWER ───")
    #print(result["answer"])
    #print("\n─── SOURCES ───")
    #for source in result["sources"]:
     #   print(source)