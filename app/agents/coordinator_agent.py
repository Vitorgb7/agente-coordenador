from app.agents.specialist_math import math_agent
from app.agents.specialist_translate import translate_agent
from app.config.settings import GROQ_API_KEY
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage


class CoordinatorAgent:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model="llama3-70b-8192"
        )

    def handle_task(self, task: str):
        decision_prompt = f"""
        Você é um coordenador de especialistas.
        Analise a tarefa abaixo e responda APENAS com uma das opções:
        - "translate" para tradução de idiomas.
        - "math" para cálculos matemáticos.
        - "none" se não houver especialista disponível.

        Tarefa: {task}
        """

        decision = self.llm.invoke([HumanMessage(content=decision_prompt)])
        decision_text = decision.content.strip().lower()

        if decision_text == "translate":
            return translate_agent(task)
        elif decision_text == "math":
            return math_agent(task)
        else:
            return "Não encontrei um especialista adequado para essa tarefa."
