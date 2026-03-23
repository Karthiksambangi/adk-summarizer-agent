import os, json, logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as adk_types
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

def summarize_text(text: str, max_sentences: int = 3) -> dict:
    return {
        "original_word_count": len(text.split()),
        "max_sentences_requested": max_sentences,
        "instruction": f"Summarize in {max_sentences} sentences:\n\n{text}"
    }

SYSTEM_PROMPT = """
You are a helpful text summarization assistant.
When given text to summarize, call the summarize_text tool, then return ONLY this JSON:
{
  "summary": "<summary here>",
  "original_word_count": <number>,
  "summary_word_count": <number>,
  "model": "gemini"
}
"""

summarizer_agent = Agent(name="summarizer_agent", model=MODEL, description="Summarizes text.", instruction=SYSTEM_PROMPT, tools=[summarize_text])
session_service = InMemorySessionService()
runner = Runner(agent=summarizer_agent, app_name="adk-summarizer", session_service=session_service)
app = FastAPI(title="ADK Summarizer Agent")

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}

@app.post("/summarize")
async def summarize(request: Request):
    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="`text` field is required.")
    max_sentences = max(1, min(int(body.get("max_sentences", 3)), 10))
    session = await session_service.create_session(app_name="adk-summarizer", user_id="user")
    content = adk_types.Content(role="user", parts=[adk_types.Part(text=f"Summarize in {max_sentences} sentences:\n\n{text}")])
    final_text = ""
    async for event in runner.run_async(user_id="user", session_id=session.id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text.strip()
    try:
        result = json.loads(final_text.replace("```json","").replace("```","").strip())
    except:
        result = {"summary": final_text, "original_word_count": len(text.split()), "summary_word_count": len(final_text.split()), "model": MODEL}
    return JSONResponse(content=result)

@app.get("/")
async def root():
    return {"agent": "ADK Summarizer Agent", "model": MODEL}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
