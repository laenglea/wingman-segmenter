import json
from typing import Optional
from fastapi import FastAPI
from semantic_text_splitter import TextSplitter
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="LLM Platform Segmenter"
)

class SegmentRequest(BaseModel):
    content: str

    max_chunk_length: Optional[int] = None

@app.post("/segment")
@app.post("/v1/segment")
async def segment(request: SegmentRequest):
    content = request.content

    capacity = 1000

    if request.max_chunk_length is not None:
        capacity = request.max_chunk_length
    
    splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", capacity)
    chunks = splitter.chunks(content)

    return {
        "chunks": chunks
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)