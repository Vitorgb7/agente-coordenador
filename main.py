from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.agents.coordinator_agent import CoordinatorAgent

app = FastAPI()

coordinator = CoordinatorAgent()

class TaskRequest(BaseModel):
    task: str

@app.post("/solve")
async def solve_task(request: TaskRequest):
    try:
        result = coordinator.handle_task(request.task)

        if hasattr(result, "content"):
            clean_result = result.content.strip()

            undesired_prefixes = [
                "Fácil!", 
                "The translation is:", 
                "Tradução:", 
                "Translation:"
            ]
            for prefix in undesired_prefixes:
                if clean_result.startswith(prefix):
                    clean_result = clean_result[len(prefix):].strip()

            return {"result": clean_result}

        return {"result": str(result).strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Agente coordenador está funcionando!"}