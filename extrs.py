import os
from typing import Tuple
import openai
from dotenv import load_dotenv
import torch
from transformers import AutoModel, AutoTokenizer  # Import from transformers

load_dotenv()

groq_api_key = os.environ.get("GROQ_API_KEY")
print(os.environ.get("OPENAI_API_KEY"))

# Load environment variables
viktor_app_secret: str = os.environ.get("VIKTOR_APP_SECRET", "")
if not viktor_app_secret:
    raise ValueError("VIKTOR_APP_SECRET is not set.")

# Split the secret into its components
API_KEY, ENDPOINT, DEPLOYMENT_NAME, EMBEDDING_DEPLOYMENT_NAME, API_VERSION = viktor_app_secret.split("|")

# Print to verify values
print(f"API_KEY: {API_KEY}")
print(f"ENDPOINT: {ENDPOINT}")

# Azure OpenAI setup
USE_AZURE = False

if viktor_app_secret:
    try:
        openai.api_type = "azure"
        openai.api_base = f"https://{ENDPOINT}.openai.azure.com/"
        openai.api_version = API_VERSION
        openai.api_key = API_KEY
        USE_AZURE = True
        print("Using Azure OpenAI")
    except ValueError:
        print("VIKTOR_APP_SECRET is not in the correct format for Azure OpenAI.")

# Check if API key and base URL are set
if openai.api_key is None or openai.api_base is None:
    raise ValueError("API key or base URL is not set.")

# Path to your local model file (Hugging Face model)
model_path = 'routellm\\routers\\matrix_factorization'  # Update this line


# Import the controller
from routellm.controller import Controller

# Create the controller
# I've used the prefix azure/ according to the LiteLLM docs
# https://litellm.vercel.app/docs/providers/azure
client = Controller(
    routers = ["mf"],
    strong_model = "azure/gpt-4o-mini",
    weak_model = "azure/gpt-4o"
)

# Make a request
try:
    response = client.chat.completions.create(
        model = "router-mf-0.11593",
        messages = [
            {"role":"user", "content":"Hello!"}
        ]
    )
    # AI Message
    message = response.choices[0].message.content
    # Model used
    model_used = response.model

    print(f"Model used: {model_used}")
    print(f"Response: {message}")

except Exception as e:
    print(f"An error occurred: {e}")