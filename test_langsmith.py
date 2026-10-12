import os
from dotenv import load_dotenv
from langsmith import Client

load_dotenv(override=True)

key = os.getenv("LANGSMITH_API_KEY")
workspace = os.getenv("LANGSMITH_WORKSPACE_ID")
endpoint = os.getenv(
    "LANGSMITH_ENDPOINT",
    "https://api.smith.langchain.com"
)

print("Key detected:", bool(key))
print("Key prefix:", key[:8] if key else None)
print("Workspace:", workspace)
print("Endpoint:", endpoint)

client = Client(
    api_key=key,
    api_url=endpoint,
    workspace_id=workspace,
)

print("Testing LangSmith...")

datasets = list(client.list_datasets(limit=1))
print("SUCCESS: LangSmith connection works.")
print("Datasets:", datasets)
from langsmith import Client
from dotenv import load_dotenv
import os

load_dotenv(override=True)

client = Client(
    api_key=os.getenv("LANGSMITH_API_KEY"),
    api_url=os.getenv("LANGSMITH_ENDPOINT"),
    workspace_id=os.getenv("LANGSMITH_WORKSPACE_ID"),
)

dataset = client.create_dataset(
    dataset_name="FANETs-test"
)

print("Created:", dataset.id)