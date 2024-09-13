from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import react_agent  # Mevcut dosyanızın adını buraya yazın

app = FastAPI()

# Question modeline max_iterations ekleniyor
class Question(BaseModel):
    question: str
    max_iterations: int = 3  # Varsayılan olarak 3 ayarlanmış

@app.post("/ask")
async def ask_question(question: Question):
    try:
        # max_iterations API isteğiyle birlikte react_agent'a iletiliyor
        answer = react_agent(question.question, max_iterations=question.max_iterations)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) # 8000 portunda local ipde açar, request ile istek atabilirsin
