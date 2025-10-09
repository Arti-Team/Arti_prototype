# LangGraph Deployment Guide

## Project Overview
Emotion-based art recommendation system - a conversational AI docent built with LangGraph that analyzes user emotions and recommends personalized artworks.

## Architecture Overview

### Core Components
- **LangGraph Orchestration**: `emotional_art_graph.py` - Main workflow controller
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

## Quick Start

### 1. Environment Setup
Create `.env` file in project root:
```bash
# Required API Keys
OPENAI_API_KEY=your-openai-key
FIREWORKS_API_KEY=your-fireworks-key  # Optional alternative
OPENAI_BASE_URL=https://api.openai.com/v1

# HuggingFace Cache Configuration
HF_HOME=/path/to/.cache/huggingface
TRANSFORMERS_CACHE=/path/to/.cache/huggingface/transformers
HUGGINGFACE_HUB_CACHE=/path/to/.cache/huggingface/hub
```

### 2. Development Server
```bash
cd arti_llm
langgraph dev --allow-blocking
```
Server runs on `http://localhost:2024`

### 3. LangGraph Workflow
```
User Input → Conversation Node → Art Curation Node → Response
              ↓                    ↓
         Memory Updates      3-Stage Pipeline
```

## API Integration

### Input Format
```json
{
  "messages": [
    {"role": "user", "content": "I'm feeling stressed about work..."}
  ]
}
```

### Output Format
```json
{
  "messages": [
    {"role": "assistant", "content": "I understand you're feeling stressed..."},
    {"role": "assistant", "content": "Here are some calming artworks..."}
  ],
  "curation_results": {
    "recommended_artworks": ["artwork_id_1", "artwork_id_2"],
    "total_candidates": 15,
    "processing_time": 3.2
  }
}
```

## Backend Implementation

### Core Endpoints
- `GET /health` - System health check
- `POST /invoke` - Main conversation endpoint
- `GET /artworks/{id}` - Artwork metadata retrieval

### Key Components
1. **Emotion Detection**: Extracts emotional state from conversation
2. **Memory Management**: Persistent user profiles and preferences  
3. **Art Recommendation**: Multi-stage filtering and ranking
4. **Response Generation**: Contextual explanations with artwork details

### Performance Notes
- Average response time: 2-4 seconds
- Concurrent user limit: 10 (adjustable)
- Cache hit rate: ~80% for repeated queries

## Frontend Integration

### WebSocket Support
```javascript
const ws = new WebSocket('ws://localhost:2024/ws');
ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  // Handle streaming responses
};
```

### Required UI Components
- Chat interface for conversation
- Artwork gallery for recommendations
- User emotion/mood selector (optional)
- Progress indicators for processing states

## Deployment Considerations

### Production Environment
- Set `OPENAI_BASE_URL` to production API endpoint
- Configure proper cache directories with sufficient storage
- Set up monitoring for API rate limits
- Implement user session management

### Scaling Options
- Horizontal: Multiple LangGraph instances with load balancer
- Vertical: Increase memory allocation for vector indices
- Caching: Redis for session and artwork metadata storage