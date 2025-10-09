# LangGraph FastAPI Deployment Guide

# 🎯 Goal

> Wrap your completed **LangGraph emotional art docent** in a **FastAPI backend**, so both a web frontend (Next.js) and mobile app (React Native / Flutter) can reuse the same API seamlessly.

## Project Overview
Emotion-based art recommendation system - a conversational AI docent built with LangGraph that analyzes user emotions and recommends personalized artworks. This guide shows how to deploy it as a production-ready FastAPI service.

## Architecture Overview

### Core Components
- **LangGraph Orchestration**: `emotional_art_graph.py` - Main workflow controller
- **FastAPI Backend**: RESTful API wrapper for cross-platform access
- **Art Curation Engine**: 3-stage recommendation pipeline (RAG → Stage A → Step 6)
- **Memory System**: User profile, preferences, and conversation history
- **RAG System**: Research paper knowledge base for art therapy

### Project Structure
```
arti_llm/
├── emotional_art_graph.py          # Main LangGraph workflow
├── rag_session.py                  # RAG knowledge system  
├── langgraph.json                  # LangGraph configuration
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
├── .gitignore                      # Git ignore rules
├── README.md                       # Detailed documentation
├── CLAUDE_EN.md                    # Development guide (English)
├── LangGraph_Deployment_Guide_EN.md # This deployment guide
├── test_phase2_integration_async.ipynb # Integration tests
└── art_curation_engine/            # Core recommendation engine
    ├── .env.example                # Engine environment template
    ├── README.md                   # Engine documentation
    ├── build_clip_index.py         # CLIP index builder
    ├── core/                       # Pipeline components
    │   ├── __init__.py
    │   ├── dynamic_stage_a.py       # Dynamic candidate collection
    │   ├── langchain_rag_system.py  # RAG system implementation
    │   ├── llm_prompts.py          # LLM prompt templates
    │   ├── rag_session_langchain.py # RAG session management
    │   ├── stage_a_candidate_collection.py # Stage A pipeline
    │   └── step6_llm_reranking.py   # Step 6 reranking
    ├── data/                       # Artwork metadata and rules
    │   ├── aic_sample/             # Sample artwork data
    │   ├── corpus/                 # Text corpus
    │   ├── rag_pdf_description.md  # RAG data description
    │   └── rules/                  # Curation rules (safety, seed)
    ├── docs/                       # Engine documentation
    │   └── SETUP.md                # Setup instructions
    ├── indices/                    # FAISS search indices
    ├── langchain_vectorstore/      # LangChain vector storage
    ├── models/                     # ML models directory
    ├── storage/                    # Cache and results
    └── tests/                      # Test suites
```

# 🧩 Recommended FastAPI Project Structure

```bash
📦 arti_llm_backend/
 ┣ 📁 graph/                       # Your LangGraph module
 ┃ ┣ __init__.py
 ┃ ┗ emotional_art_graph.py        # Completed LangGraph graph
 ┣ 📁 api/
 ┃ ┣ __init__.py
 ┃ ┣ routes_chat.py                # Main chat endpoint
 ┃ ┣ routes_artworks.py            # Artwork metadata endpoints
 ┃ ┗ routes_memory.py              # User memory endpoints
 ┣ 📁 core/
 ┃ ┣ config.py                     # Environment configs
 ┃ ┗ dependencies.py               # FastAPI dependencies
 ┣ 📁 models/
 ┃ ┣ schema.py                     # Request / Response models
 ┃ ┗ memory.py                     # Memory data models
 ┣ 📁 art_curation_engine/         # Existing curation engine
 ┃ ┗ ... (all existing files)
 ┣ main.py                         # FastAPI entry point
 ┣ requirements.txt                # Updated with FastAPI deps
 ┗ README.md
```

This modular structure makes it easy to maintain, test, and scale across web and mobile platforms.

---

# ⚙️ Dependencies (`requirements.txt`)

Add FastAPI dependencies to your existing requirements:

```txt
# Existing LangGraph dependencies
langgraph
langchain
langchain-openai
trustcall
python-dotenv
faiss-cpu
sentence-transformers

# FastAPI additions
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
python-multipart
httpx

# Optional for production
gunicorn
redis
psycopg2-binary
```

---

# 🧠 Wrap Your LangGraph (`graph/emotional_art_graph.py`)

Create a **builder function** so FastAPI can import and reuse your graph:

```python
# graph/emotional_art_graph.py
import os
from typing import Dict, Any
from langgraph.graph import StateGraph
from dotenv import load_dotenv

# Import your existing graph components
from .conversation_node import conversation_node
from .art_curation_node import art_curation_node

load_dotenv()

def build_emotional_art_graph():
    """
    Build and return the emotional art docent graph
    """
    # Create the graph (using your existing implementation)
    workflow = StateGraph(EmotionalArtState)
    
    # Add nodes
    workflow.add_node("conversation_node", conversation_node)
    workflow.add_node("art_curation_node", art_curation_node)
    
    # Add edges and conditional routing
    workflow.set_entry_point("conversation_node")
    workflow.add_conditional_edges(
        "conversation_node",
        should_curate_art,
        {
            "curate": "art_curation_node",
            "continue": END
        }
    )
    workflow.add_edge("art_curation_node", END)
    
    return workflow.compile()

# Initialize graph instance
_graph_instance = None

def get_graph():
    """Get or create graph instance (singleton pattern)"""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_emotional_art_graph()
    return _graph_instance
```

---

# 🌐 Create FastAPI Routes

## Main Chat Endpoint (`api/routes_chat.py`)

```python
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
from models.schema import ChatRequest, ChatResponse, ArtRecommendation
from graph.emotional_art_graph import get_graph
from core.dependencies import get_user_session

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    session_id: str = Depends(get_user_session)
):
    """
    Main chat endpoint for emotional art docent
    """
    try:
        graph = get_graph()
        
        # Prepare input with session context
        graph_input = {
            "messages": request.messages,
            "session_id": session_id,
            "user_context": request.user_context or {}
        }
        
        # Invoke the graph
        result = await graph.ainvoke(graph_input)
        
        # Extract response components
        response_messages = result.get("messages", [])
        curation_results = result.get("curation_results", {})
        memory_updates = result.get("memory_updates", {})
        
        return ChatResponse(
            messages=response_messages,
            curation_results=curation_results,
            memory_updates=memory_updates,
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "arti-llm-backend"}
```

## Artwork Endpoints (`api/routes_artworks.py`)

```python
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from models.schema import ArtworkDetail, ArtworkSearch

router = APIRouter()

@router.get("/artworks/{artwork_id}", response_model=ArtworkDetail)
async def get_artwork(artwork_id: str):
    """Get detailed artwork information"""
    # Implementation using your art_curation_engine
    pass

@router.post("/artworks/search", response_model=List[ArtworkDetail])
async def search_artworks(search: ArtworkSearch):
    """Search artworks by criteria"""
    # Implementation using your Stage A system
    pass
```

---

# 🧾 Request/Response Models (`models/schema.py`)

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class Message(BaseModel):
    role: str = Field(..., description="Message role: user, assistant, system")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="Conversation messages")
    user_context: Optional[Dict[str, Any]] = Field(None, description="User context data")
    
class ArtRecommendation(BaseModel):
    artwork_id: str
    title: str
    artist: str
    score: float
    justification: str
    image_url: Optional[str] = None

class CurationResults(BaseModel):
    recommended_artworks: List[ArtRecommendation]
    total_candidates: int
    processing_time: float
    emotion_analysis: Dict[str, Any]

class ChatResponse(BaseModel):
    messages: List[Message]
    curation_results: Optional[CurationResults] = None
    memory_updates: Optional[Dict[str, Any]] = None
    session_id: str

class ArtworkDetail(BaseModel):
    artwork_id: str
    title: str
    artist: str
    description: str
    metadata: Dict[str, Any]
    emotional_tags: List[str]

class ArtworkSearch(BaseModel):
    query: Optional[str] = None
    emotions: Optional[List[str]] = None
    limit: int = Field(10, ge=1, le=100)
```

---

# 🚀 Main FastAPI Application (`main.py`)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from api.routes_chat import router as chat_router
from api.routes_artworks import router as artwork_router
from core.config import settings

# Create FastAPI app
app = FastAPI(
    title="ArtiTech Emotional Art Docent API",
    description="FastAPI backend for LangGraph emotional art recommendation system",
    version="1.0.0"
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
app.include_router(artwork_router, prefix="/api/v1", tags=["artworks"])

@app.get("/")
def root():
    return {
        "message": "ArtiTech LangGraph Emotional Art Docent API", 
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

# ⚙️ Configuration (`core/config.py`)

```python
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ArtiTech Emotional Art Docent"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    # LangGraph Settings
    OPENAI_API_KEY: str
    FIREWORKS_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    
    # HuggingFace
    HF_HOME: str = "/tmp/.cache/huggingface"
    TRANSFORMERS_CACHE: str = "/tmp/.cache/huggingface/transformers"
    HUGGINGFACE_HUB_CACHE: str = "/tmp/.cache/huggingface/hub"
    
    # Performance
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 30
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

# 🏃‍♂️ Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-key-here"

# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open 👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**
You'll see interactive Swagger UI to test all endpoints.

---

# 💬 Example Frontend Integration

## Web Request (Next.js)
```javascript
const sendMessage = async (message) => {
  const response = await fetch("http://localhost:8000/api/v1/chat", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      messages: [{ role: "user", content: message }]
    }),
  });
  
  const data = await response.json();
  console.log("AI Response:", data.messages);
  console.log("Art Recommendations:", data.curation_results);
};
```

## Mobile Request (React Native)
```javascript
const chatWithAI = async (userMessage) => {
  try {
    const response = await fetch("https://your-api-domain.com/api/v1/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        messages: [{ role: "user", content: userMessage }],
        user_context: { platform: "mobile" }
      }),
    });
    
    const result = await response.json();
    return result;
  } catch (error) {
    console.error("Chat API error:", error);
  }
};
```

---

# 🚢 Deployment Options

## Option 1 — Render (Simplest)
```bash
# Add to requirements.txt
gunicorn>=21.0.0

# Create start script
echo "gunicorn -w 2 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT" > start.sh
```

## Option 2 — Docker (Portable)
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Option 3 — AWS Lambda (Serverless)
```python
# Add to main.py
from mangum import Mangum
handler = Mangum(app)
```

---

# 📱 Future Expansion Roadmap

| Phase             | Goal                               | Implementation                    |
| ----------------- | ---------------------------------- | --------------------------------- |
| ✅ **Current**     | FastAPI wrapper for LangGraph      | `/chat` endpoint with full features |
| 🌐 **Web Phase**   | Connect Next.js frontend           | WebSocket support, session management |
| 📱 **Mobile Phase** | React Native/Flutter app          | Mobile-optimized responses         |
| 💾 **Data Phase**  | Add PostgreSQL/Redis               | Persistent memory, analytics       |
| 🤖 **AI Phase**    | Enhanced multimodal capabilities   | Vision models, voice integration   |

---

# ✅ Key Benefits of FastAPI Approach

* **Cross-Platform**: Same API works for web, mobile, and future integrations
* **Performance**: Async support handles concurrent art curation requests efficiently  
* **Documentation**: Auto-generated OpenAPI docs for frontend teams
* **Scalability**: Easy horizontal scaling with load balancers
* **Type Safety**: Pydantic models ensure data validation
* **Production Ready**: Built-in security, CORS, and monitoring support

Your existing LangGraph implementation remains unchanged - FastAPI simply provides a robust API layer for multi-platform access.