import os
from dotenv import load_dotenv  # 1. Import dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

def main():
    # 2. Load your environment variables (API keys)
    load_dotenv()
    
    file_path = r"C:\LangChain_Super\langchain_work\mediumblog1.txt"
    
    # 3. Read the text file using standard Python
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    # 4. Initialize the CharacterTextSplitter
    # Configure chunk sizes and your preferred separator sequence
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # 5. Split the text and turn chunks into Document objects instantly
    # We pass the metadata dictionary directly so every chunk knows its file origin
    final_chunks = text_splitter.create_documents(
        texts=[raw_text], 
        metadatas=[{"source": file_path}]
    )
    
    print(f"Successfully split text file into {len(final_chunks)} chunks.")
    
    # Quick debug preview of your first chunk
    if final_chunks:
        print("\n--- First Chunk Preview ---")
        print(final_chunks[0].page_content[:200])

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    print("Embeddings model initialized successfully.")
    PineconeVectorStore.from_documents(
        documents=final_chunks,
        embedding=embeddings,
        index_name=os.environ['INDEX_NAME']
    )
print("Documents successfully added to Pinecone index.")

if __name__ == "__main__":
    main()
