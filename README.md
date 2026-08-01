LangChain Leave Policy Bot
An AI-powered document Q&A web application built using **LangChain**, **OpenAI**, **Hugging Face**, **ChromaDB / FAISS**. This application ingests company policy PDFs (such as `Leave-policy.pdf`), chunks and embeds document text, and enables interactive Q&A through an intuitive web UI.
Features
- **PDF Ingestion:** Extracts text from document files using `PyPDFLoader`.
- **Text Chunking:** Utilizes `RecursiveCharacterTextSplitter` to optimize context window performance and maintain semantic structure.
- **Flexible Embeddings:** Integrates both cloud-based OpenAI embeddings and free, local Hugging Face models (`sentence-transformers/all-MiniLM-L6-v2`).
- **Vector Storage:** Indexes and queries document embeddings via Chroma DB or FAISS vector stores.
