import re
import uvicorn

from typing import Optional
from fastapi import FastAPI, Response
from pydantic import BaseModel
from semantic_text_splitter import MarkdownSplitter, TextSplitter

app = FastAPI(
    title="LLM Platform Segmenter"
)

class SegmentRequest(BaseModel):
    content: str

    max_chunk_length: Optional[int] = None

@app.get("/")
def segment_post(content: str = ""):
    if not content:
        return Response(content="LLM Platform Segmenter", media_type="text/html")
    
    return segment(content)

@app.post("/")
def segment_post(request: SegmentRequest):
    return segment(request.content,  request.max_chunk_length)

def segment(content: str, capacity: Optional[int] = None):
    if capacity is None:
        capacity = 1000

    if is_markdown(content):
        splitter = MarkdownSplitter(capacity)
        chunks = splitter.chunks(content)
    else:
        splitter = TextSplitter(capacity)
        chunks = splitter.chunks(content) 

    return {
        "chunks": chunks
    }

def is_markdown(text):
    patterns = [
        r'^#{1,6}\s',                    # Headings
        r'\*\*[^*]+\*\*',                # Bold with **
        r'__[^_]+__',                    # Bold with __
        r'\*[^*]+\*',                    # Italic with *
        r'_[^_]+_',                      # Italic with _
        r'\[.+?\]\(.+?\)',               # Links
        r'!\[.+?\]\(.+?\)',              # Images
        r'^\s*[-\*\+]\s+',               # Unordered lists
        r'^\s*\d+\.\s+',                 # Ordered lists
        r'^>\s+',                        # Blockquotes
        r'`{1,3}[^`]+`{1,3}',            # Inline code or code blocks
        r'^\s*---+\s*$',                 # Horizontal rules
    ]
    
    return bool(re.search('|'.join(patterns), text, flags=re.MULTILINE))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)