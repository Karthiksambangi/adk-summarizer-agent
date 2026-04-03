import os, json, logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from groq import Groq

logging.basicConfig(level=logging.INFO)
client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

app = FastAPI(title="ADK Summarizer Agent")

@app.get("/health")
async def health():
    return {"status": "ok", "model": "llama-3.3-70b"}

@app.post("/summarize")
async def summarize_endpoint(request: Request):
    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="`text` field is required.")
    max_sentences = max(1, min(int(body.get("max_sentences", 3)), 10))
    try:
        prompt = f"Summarize the following text in {max_sentences} sentences. Return ONLY this JSON format with no extra text:\n{{\"summary\": \"your summary\", \"original_word_count\": {len(text.split())}, \"summary_word_count\": 0, \"model\": \"llama\"}}\n\nTEXT: {text}"
        response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.3)
        result_text = response.choices[0].message.content.strip()
        cleaned = result_text.replace("```json","").replace("```","").strip()
        result = json.loads(cleaned)
        result["summary_word_count"] = len(result.get("summary","").split())
    except Exception as e:
        result = {"summary": str(e), "original_word_count": len(text.split()), "summary_word_count": 0, "model": "llama"}
    return JSONResponse(content=result)

@app.get("/")
async def root():
    return {"agent": "ADK Summarizer Agent", "model": "llama-3.3-70b", "track": "Track 1"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
