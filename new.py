from openai import OpenAI
import os
from dotenv import load_dotenv
from config import OPENAI_API_KEY
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

for model in client.models.list():
    print(model.id)