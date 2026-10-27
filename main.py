import os
from dotenv import load_dotenv  
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# 1. Load the .env file variables into os.environ
load_dotenv()  
print("Environment variables loaded successfully.")

# Initialize OpenAI Embeddings and LLM
# Note: LangChain automatically looks for OPENAI_API_KEY in os.environ, 
# so passing it explicitly is optional but fine.
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
llm = ChatOpenAI(model="gpt-5.6-terra")

# Initialize Pinecone Vector Store
# Ensure PINECONE_API_KEY is in your .env file
vectorstore = PineconeVectorStore(
    embedding=embeddings,
    index_name=os.environ['INDEX_NAME']
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
{context}
Question: {question}

Provide a detailed answer:
    """
)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieve_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.
    """
    docs = retriever.invoke(query)
    context = format_docs(docs)
    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)
    return response.content

# FIX: Moved to the root level (unindented) so it actually runs
if __name__ == "__main__":
    print("Starting the retrieval process...")

    query = "What is Pinecone in machine learning?"
    response = retrieve_chain_without_lcel(query)
    print("\nResponse:")
    print(response)
