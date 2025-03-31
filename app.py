import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage
from fastapi import FastAPI
from pydantic import BaseModel


# Load environment variables
load_dotenv()

# Ensure API key is loaded
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Error: API Key is missing! Please set GOOGLE_API_KEY in .env file.")
    exit(1)

# Initialize chat model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

# Define a prompt template
prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template="You are a helpful chatbot. Answer the following: {user_input}"
)

# Create an LLM Chain
chat_chain = LLMChain(
    llm=model,
    prompt=prompt_template
)

def chatbot_response(user_input):
    """ Get response from AI model using LLM Chain """
    try:
        response = chat_chain.invoke({"user_input": user_input})
        return response["text"]
    except Exception as e:
        return f"Error: {str(e)}"

app = FastAPI()

# Define request body
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    """ Receive user input and return chatbot response """
    try:
        response = model.invoke([HumanMessage(content=request.message)])
        return {"response": response.content}
    except Exception as e:
        return {"error": str(e)}
