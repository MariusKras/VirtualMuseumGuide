import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX"])
namespace = os.getenv("PINECONE_NAMESPACE", "default")

query = input("Enter your query: ")

r = client.embeddings.create(model="text-embedding-3-small", input=query)
vec = r.data[0].embedding

res = index.query(
    namespace=namespace,
    vector=vec,
    top_k=3,
    include_values=False,
    include_metadata=True,
)

for match in res.matches:
    print(f"Score: {match.score:.3f}")
    print(f"ID: {match.id}")
    print("-" * 40)
