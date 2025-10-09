# CLAUDE Development Guide

## Project Status

### Completed Phases
- ✅ **Phase 1**: Core Conversation Architecture
- ✅ **Phase 2**: Art Curation Integration 
- ✅ **Phase 3**: Memory System

### Current Status (2025-10-09)
**All major features are fully operational.**

## Recent Modifications

### Art Curation Node Data Format Fix
- **Issue**: AttributeError occurred when Stage A → Step 6 data format mismatch
- **Solution**: Implemented candidate ID conversion logic in `emotional_art_graph.py`
- **Result**: Complete functionality in LangGraph Studio

### Architecture Changes
- **Previous**: Single `emotional_art_docent` node
- **Current**: Separated `conversation_node` + `art_curation_node`
- **Routing**: Conditional edge `should_curate_art()` for node transitions

## Execution Methods

### LangGraph Studio
```bash
cd /Users/ijunhyeong/Desktop/arti_llm
source .env
export HF_HOME=/Users/ijunhyeong/.cache/huggingface
export TRANSFORMERS_CACHE=/Users/ijunhyeong/.cache/huggingface/transformers
export HUGGINGFACE_HUB_CACHE=/Users/ijunhyeong/.cache/huggingface/hub
langgraph dev --allow-blocking
```

### Testing Environment
- **Notebook**: `test_phase2_integration_async.ipynb`
- **Studio URL**: http://localhost:2123

## Core Components

### 1. Emotion Extraction System
- `extract_emotion_hints()`: Extract emotional hints from conversation
- `extract_situation_hints()`: Extract situational context
- Automatic memory update triggers

### 2. Memory System
- **Profile**: User demographics and preferences
- **Art Preferences**: Art preference patterns
- **Interaction Patterns**: Conversation behavior analysis
- `ThreadSafeMemoryProcessor` implementation

### 3. Art Curation Engine
- **Step 5**: RAG-based brief generation
- **Stage A**: Candidate collection (CLIP + text similarity)
- **Step 6**: LLM-based reranking (final 8 selections)

### 4. Asynchronous Processing
- `asyncio.wait_for()`: Timeout management
- `loop.run_in_executor()`: Async execution of sync functions
- Thread-safe memory processing

## Important Files

### Core Files
- `emotional_art_graph.py`: Main LangGraph architecture
- `.env`: Environment variables (including HuggingFace cache paths)

### Art Curation Engine
- `art_curation_engine/core/langchain_rag_system.py`: RAG system
- `art_curation_engine/core/stage_a_candidate_collection.py`: Candidate collection
- `art_curation_engine/core/step6_llm_reranking.py`: LLM reranking

### Test Files
- `test_phase2_integration_async.ipynb`: Integration testing

## Environment Configuration

### Required Environment Variables
```bash
# API Keys
OPENAI_API_KEY=your-api-key
FIREWORKS_API_KEY=your-fireworks-key
OPENAI_BASE_URL=https://api.fireworks.ai/inference/v1

# HuggingFace Cache (X31 warning resolution)
HF_HOME=/Users/ijunhyeong/.cache/huggingface
TRANSFORMERS_CACHE=/Users/ijunhyeong/.cache/huggingface/transformers
HUGGINGFACE_HUB_CACHE=/Users/ijunhyeong/.cache/huggingface/hub
```

## Debugging Guide

### Common Issues
1. **HuggingFace Path Errors**: Check environment variables
2. **Store is None**: Automatic InMemoryStore fallback implemented
3. **Async/Await Errors**: Verify `loop.run_in_executor` pattern
4. **Data Format Errors**: Check Stage A → Step 6 conversion logic

### Log Monitoring
- Check DEBUG messages in LangGraph Studio console
- Monitor `art_curation_node` step-by-step progress

## Next Steps (Future Development)
- Memory optimization
- Recommendation accuracy improvement
- New art database integration
- User interface enhancements