import os
import uuid
import chromadb
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# 1. DATA => DOCUMENTS
def load_all_pdfs(folder_path="."):
    """Loads all PDF documents from the specified directory."""
    num_docs = 0
    all_docs = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            
            loader = PyPDFLoader(pdf_path)
            doc = loader.load()
            
            all_docs.extend(doc)
            num_docs += 1
            
    print("total pdfs:", num_docs)
    print("total pages:", len(all_docs))
    
    return all_docs



# 2. CHUNKS
def split_document(documents, chunk_size=500, chunk_overlap=50):
    """Splits documents into smaller chunks for embeddings."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs



# 3. EMBEDDINGS
class EmbeddingsManager():
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        print("loading model....", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        print("embeddings dimensions", self.model.get_sentence_embedding_dimension())
        
    def generate_embeddings(self, text):
        embeddings = self.model.encode(text, show_progress_bar=True)
        print("Embeddings Shape: ", embeddings.shape)
        return embeddings



# 4. VECTOR STORE
class VectorManager():
    def __init__(self, persist_directory="vector_store", collection_name="pdf_documents"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.collection = None
        self.client = None

    def initialize_store(self):
        os.makedirs(self.persist_directory, exist_ok=True)

        # create a client
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # create the collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "vector store collection for pdf embeddings in RAG"}
        )

        print("initialize the vector with collections:", self.collection_name)
        print("docs in collection:", self.collection.count())

    def add_documents(self, documents, embeddings):
        if len(documents) != len(embeddings):
            raise ValueError("number of documents does not match number of embeddings")

        ids = []
        all_metadata = []
        documents_content = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4()}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)

            all_metadata.append(metadata)
            documents_content.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        # Batch insert into ChromaDB
        self.collection.add(
            ids=ids,
            metadatas=all_metadata,
            documents=documents_content,
            embeddings=embeddings_list
        )

        print("total documents added in vector store:", len(documents_content))
        print("docs in collection:", self.collection.count()) 


# 5. RETRIEVAL PIPELINE
class RAG_Retriever:
    def __init__(self, embedding_manager, vector_store):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def retrieve(self, query, top_k=5, score_threshold=0.0):
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        retrieved_docs = []

        if results.get("documents") and results["documents"][0]:
            res_ids = results["ids"][0]
            res_metadatas = results["metadatas"][0]
            res_documents = results["documents"][0]
            res_distances = results["distances"][0]

            for i, (doc_id, meta, doc_text, distance) in enumerate(zip(res_ids, res_metadatas, res_documents, res_distances)):
                similarity_score = 1 - distance

                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        "id": doc_id,
                        "document": doc_text,
                        "metadata": meta,
                        "distance": distance,
                        "similarity_score": similarity_score,
                        "rank": i + 1
                    })
        else:
            print("no documents found")

        return retrieved_docs



# 6. MAIN EXECUTION (PIPELINE RUN)
if __name__ == "__main__":
    print("--- Starting RAG Pipeline ---")
    
    # 1. Load PDFs
    all_pdf_documents = load_all_pdfs(".") 
    
    if len(all_pdf_documents) == 0:
        print("No PDFs found in the current directory. Please add a PDF and try again.")
    else:
        # 2. Chunk Documents
        chunks = split_document(all_pdf_documents)
        print("Total chunks created:", len(chunks))
        
        # 3. Initialize Embeddings Manager
        embedding_manager = EmbeddingsManager()
        
        # 4. Generate Embeddings for chunks
        texts = [doc.page_content for doc in chunks]
        chunk_embeddings = embedding_manager.generate_embeddings(texts)
        
        # 5. Initialize Vector Store and Add Documents
        vector_store = VectorManager()
        vector_store.initialize_store()
        vector_store.add_documents(chunks, chunk_embeddings)
        
        # 6. Initialize Retriever and Query
        rag_retriever = RAG_Retriever(embedding_manager=embedding_manager, vector_store=vector_store)
        
        test_query = "what is RAG?"
        print(f"\n--- Querying: '{test_query}' ---")
        query_results = rag_retriever.retrieve(test_query)
        
        # 7. Print Results
        for result in query_results:
            print(f"\n[Rank {result['rank']} | Score: {result['similarity_score']:.4f}]")
            print(f"Content: {result['document'][:200]}...") # Printing first 200 chars for brevity