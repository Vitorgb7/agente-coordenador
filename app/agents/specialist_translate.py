from app.config.settings import GROQ_API_KEY
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama3-70b-8192",
    groq_api_key=GROQ_API_KEY
)

def translate_agent(task: str):
    prompt = PromptTemplate.from_template(
        "Você é um especialista em tradução. Traduza para o inglês: {task}"
    )
    chain = prompt | llm
    return chain.invoke({"task": task})
