from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# Initialize the PDF loader
loader = PyPDFLoader("Leave-policy.pdf")

# Load documents (PyPDFLoader splits pages into individual Document objects)
documents = loader.load()

#print("Number of pages/documents:", len(documents))

#print("\n--- Content of Page 1 ---")
#print(documents[0].page_content)

# 2. Initialize the Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # Max characters per chunk
    chunk_overlap=200,      # Overlap between chunks to keep context
    length_function=len,
    is_separator_regex=False,
)

# 3. Split the loaded documents into smaller chunks
chunks = text_splitter.split_documents(documents)

#print("Total chunks created:", len(chunks))
#print("\n--- Content of First Chunk ---")
#print(chunks[0].page_content)
#print("----------")
#print(chunks[1].page_content)

#4. Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#5
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Indexed successfully with local embeddings!")

# 6
query = "List out the rules for leave encashment?"
results = vectorstore.similarity_search(query, k=2)

print("\n--- Search Result ---")
print(results[0].page_content)

