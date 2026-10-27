import os
from time import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv  # 1. Import dotenv

load_dotenv()  # 2. Load the .env file variables into os.environ

# Now this will work seamlessly without throwing a KeyError
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

# Create Index if not already created
pinecone_index_name = "medium-blogs-embeddings-index-1"
if pinecone_index_name not in pc.list_indexes().names():
    pc.create_index(
        name=pinecone_index_name, 
        dimension=1536, # '1536' is the dimension for ada-002 embeddings
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
     
    while not pc.describe_index(pinecone_index_name).index.status['ready']:
        time.sleep(1)
    
    print("Pinecone Index provisioned")
else:
    print("Pinecone Index Already Provisioned")