from app.config.settings import GROQ_API_KEY
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama3-8b-8192",
    groq_api_key=GROQ_API_KEY
)

def math_agent(task: str):
    prompt = PromptTemplate.from_template(
        "Você é um agente especialista em matemática. Resolva: {task}"
    )
    chain = prompt | llm
    return chain.invoke({"task": task})
