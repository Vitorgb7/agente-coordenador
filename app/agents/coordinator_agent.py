import logging
import os
import traceback
from app.agents.specialist_math import math_agent
from app.agents.specialist_translate import translate_agent
from app.config.settings import GROQ_API_KEY
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class CoordinatorAgent:
    def __init__(self):
        logger.debug("Inicializando CoordinatorAgent...")
        logger.debug(f"GROQ_API_KEY carregada (parcial): {GROQ_API_KEY[:8]}...")

        try:
            self.llm = ChatGroq(
                groq_api_key=GROQ_API_KEY.strip(),
                model="llama3-70b-8192"
            )
            logger.debug("ChatGroq inicializado com sucesso.")
        except Exception as e:
            logger.error("Erro ao inicializar ChatGroq:")
            logger.error(traceback.format_exc())
            raise e

    def handle_task(self, task: str):
        logger.info(f"Recebida tarefa: {task}")

        decision_prompt = f"""
        Você é um coordenador de especialistas.
        Analise a tarefa abaixo e responda APENAS com uma das opções:
        - "translate" para tradução de idiomas.
        - "math" para cálculos matemáticos.
        - "none" se não houver especialista disponível.

        Tarefa: {task}
        """

        try:
            logger.debug("Enviando prompt para o modelo...")
            decision = self.llm.invoke([HumanMessage(content=decision_prompt)])
            logger.debug(f"Resposta recebida do modelo: {decision}")
        except Exception as e:
            logger.error("Erro ao invocar o modelo:")
            logger.error(traceback.format_exc())
            raise e

        decision_text = decision.content.strip().lower()
        logger.info(f"Decisão do modelo: {decision_text}")

        if decision_text == "translate":
            logger.debug("Delegando para specialist_translate...")
            return translate_agent(task)
        elif decision_text == "math":
            logger.debug("Delegando para specialist_math...")
            return math_agent(task)
        else:
            logger.warning("Nenhum especialista identificado.")
            return "Não encontrei um especialista adequado para essa tarefa."