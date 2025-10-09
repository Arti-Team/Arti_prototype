ㄷw#!/usr/bin/env python3
"""
RAG Session Brief Generator

Generates evidence-based curation briefs by querying research papers
and creating structured recommendations for artwork selection.

Usage:
    from rag_session import RAGSessionBrief
    
    brief_gen = RAGSessionBrief()
    brief = brief_gen.generate_brief("feeling stressed at work", ["anxiety", "overwhelmed"])
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LlamaIndex imports
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core import Settings

import faiss

class RAGSessionBrief:
    """
    RAG Session Brief Generator
    
    Combines research evidence with LLM reasoning to generate
    personalized curation briefs for artwork recommendation.
    """
    
    def __init__(self, index_path: str = None, cache_dir: str = None):
        # Use relative paths if no specific path provided
        script_dir = Path(__file__).parent
        self.index_path = Path(index_path) if index_path else script_dir / "indices" / "rag_faiss"
        self.model_dir = script_dir / "models" / "all-MiniLM-L6-v2"
        self.cache_dir = Path(cache_dir) if cache_dir else script_dir / ".cache" / "rag_sessions"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.query_engine = None
        self.llm = None
        self._setup_rag_system()
        self._setup_llm()
        
    def _setup_rag_system(self):
        """Load the persisted RAG index and setup query engine"""
        print("🔄 Loading RAG index...")
        
        # Setup embedding model with relative path
        embed_model = HuggingFaceEmbedding(
            model_name=str(self.model_dir)
        )
        
        # Set global settings
        Settings.embed_model = embed_model
        
        # Create empty FAISS vector store - this will be populated from storage
        faiss_index = faiss.IndexFlatL2(384)  # all-MiniLM-L6-v2 dimension
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        
        # Load storage context from persisted directory
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(self.index_path)
        )
        
        # Load the index with proper embedding model
        index = load_index_from_storage(storage_context, embed_model=embed_model)
        
        # Create query engine
        self.query_engine = index.as_query_engine(
            similarity_top_k=10,
            response_mode="no_text"  # We only want the source nodes
        )
        
        print("✅ RAG index loaded successfully")
        
    def _setup_llm(self):
        """Setup LLM for brief generation"""
        # Use Fireworks API like in test_rag_performance.py
        fireworks_key = os.getenv("FIREWORKS_API_KEY")
        
        if fireworks_key and fireworks_key != "your_fireworks_api_key_here":
            self.llm = LlamaOpenAI(
                model="gpt-4o",
                api_key=fireworks_key,
                api_base="https://api.fireworks.ai/inference/v1",
                temperature=0.1
            )
            print("✅ LLM setup complete (Fireworks)")
            return
        
        # Fallback to OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and openai_key != "your-api-key-here":
            self.llm = LlamaOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                api_key=openai_key
            )
            print("✅ LLM setup complete (OpenAI)")
            return
            
        print("⚠️ Warning: No valid API key found for LLM")
        
    def make_rag_queries(self, situation: str, emotions: List[str]) -> List[str]:
        """Generate targeted RAG queries based on user situation and emotions"""
        emotion_str = ', '.join(emotions)
        
        base_queries = [
            f"상황: {situation}. 감정: {emotion_str}. 각성 저감에 적합한 색채(명도/채도/색상) 가이드 요약",
            "Ecological Valence Theory 핵심과 파랑/초록 선호의 근거",
            "불안/분노 고각성에서 피해야 할 시각 자극(색 대비/주제) 근거", 
            "자연/블루스페이스(물가/하늘)의 정서 효과 근거",
            "미술치료/자연 이미지 중재의 효과(메타분석/리뷰) 핵심 수치"
        ]
        
        # Add emotion-specific queries
        if "anxiety" in emotions or "불안" in emotions:
            base_queries.append("불안 감소를 위한 색채 치료 연구 결과")
            base_queries.append("고채도 색상이 불안에 미치는 부정적 영향")
            
        if "stress" in emotions or "스트레스" in emotions:
            base_queries.append("스트레스 완화를 위한 자연 경관 이미지의 효과")
            base_queries.append("업무 스트레스와 색채 환경의 상관관계")
            
        if "depression" in emotions or "우울" in emotions:
            base_queries.append("우울증에 대한 미술치료 메타분석 결과")
            base_queries.append("밝은 색상과 기분 개선의 상관관계")
            
        return base_queries
    
    def fetch_evidence(self, queries: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """Fetch evidence from RAG system for given queries"""
        if not self.query_engine:
            raise RuntimeError("RAG system not initialized")
            
        print(f"🔄 Fetching evidence for {len(queries)} queries...")
        
        evidence_list = []
        for i, query in enumerate(queries):
            print(f"  Query {i+1}/{len(queries)}: {query[:50]}...")
            
            try:
                response = self.query_engine.query(query)
                
                for source_node in response.source_nodes[:top_k]:
                    metadata = source_node.node.metadata
                    
                    evidence = {
                        "title": metadata.get("title", "Unknown"),
                        "year": metadata.get("year", "Unknown"),
                        "domain": metadata.get("domain", "Unknown"),
                        "score": float(source_node.score) if source_node.score else 0.0,
                        "snippet": source_node.node.get_content()[:650].replace("\n", " "),
                        "query_used": query
                    }
                    evidence_list.append(evidence)
                    
            except Exception as e:
                print(f"    ⚠️ Error with query: {e}")
                continue
        
        # Remove duplicates based on title and snippet start
        unique_evidence = {}
        for evidence in evidence_list:
            key = (evidence["title"], evidence["snippet"][:120])
            if key not in unique_evidence:
                unique_evidence[key] = evidence
                
        final_evidence = list(unique_evidence.values())[:12]  # Limit to top 12
        print(f"✅ Collected {len(final_evidence)} unique evidence pieces")
        
        return final_evidence
    
    def render_brief_prompt(self, evidence: List[Dict[str, Any]], situation: str, emotions: List[str]) -> str:
        """Generate prompt for LLM brief generation"""
        evidence_text = "\n".join([
            f"- ({e['title']} {e.get('year')}) {e['snippet']}" 
            for e in evidence
        ])
        
        prompt = f"""당신은 근거 중심 미술 큐레이터입니다.
아래 evidence만 근거로, 오늘 사용자 상황에 맞는 세션 브리프를 JSON으로 작성하세요.

입력:
- user_situation_summary: "{situation}"
- user_emotion: {emotions}

출력(JSON):
{{
  "curatorial_goals": [ "3~5개의 큐레이션 목표" ],
  "desired_themes": [ "5~8개의 원하는 주제" ],
  "avoid_today": [ "오늘 피해야 할 요소들" ],
  "palette_direction": {{"hue": ["blue","green"], "saturation":"low", "value":"high"}},
  "style_tempo": ["impressionist","soft edges","slow tempo"],
  "citations": [{{"title":"...", "year":"..."}}]
}}

제약:
- evidence 밖 추측 금지, JSON만 반환
- curatorial_goals: 사용자 감정 상태 개선을 위한 구체적 목표
- desired_themes: water/sky/garden/companionship/nature 등 구체적 주제
- avoid_today: 현재 감정 상태에 부적절한 주제나 색상
- palette_direction: HSV 기반 색상 가이드라인
- style_tempo: 화풍과 리듬감 가이드라인
- citations: 사용된 연구 논문 출처

evidence:
<<<
{evidence_text}
>>>"""
        return prompt
    
    def call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call LLM and parse JSON response"""
        if not self.llm:
            raise RuntimeError("LLM not initialized - check OPENAI_API_KEY")
            
        try:
            response = self.llm.complete(prompt)
            response_text = response.text.strip()
            
            # Try to extract JSON from response
            if "```json" in response_text:
                # Extract JSON from code block
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif response_text.startswith("```"):
                # Remove code block markers
                response_text = response_text.strip("`").strip()
                
            # Parse JSON
            brief_json = json.loads(response_text)
            return brief_json
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Response was: {response_text[:200]}...")
            raise
        except Exception as e:
            print(f"❌ LLM call error: {e}")
            raise
    
    def _get_cache_key(self, situation: str, emotions: List[str]) -> str:
        """Generate cache key for situation and emotions"""
        content = f"{situation}:{':'.join(sorted(emotions))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load result from cache if exists"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """Save result to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    def generate_brief(self, situation: str, emotions: List[str], use_cache: bool = True) -> Dict[str, Any]:
        """
        Generate complete session brief
        
        Args:
            situation: User situation description
            emotions: List of user emotions
            use_cache: Whether to use caching
            
        Returns:
            Complete session brief with evidence and citations
        """
        print(f"\n🎯 Generating session brief...")
        print(f"   Situation: {situation}")
        print(f"   Emotions: {emotions}")
        
        # Check cache first
        cache_key = self._get_cache_key(situation, emotions)
        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached:
                print("✅ Using cached result")
                return cached
        
        # Step 1: Generate RAG queries
        queries = self.make_rag_queries(situation, emotions)
        print(f"📝 Generated {len(queries)} RAG queries")
        
        # Step 2: Fetch evidence
        evidence = self.fetch_evidence(queries)
        
        # Step 3: Generate brief prompt
        prompt = self.render_brief_prompt(evidence, situation, emotions)
        
        # Step 4: Call LLM
        print("🤖 Calling LLM to generate brief...")
        brief_json = self.call_llm(prompt)
        
        # Step 5: Package complete result
        result = {
            "brief": brief_json,
            "evidence_used": evidence,
            "queries_used": queries,
            "user_input": {
                "situation": situation,
                "emotions": emotions
            },
            "metadata": {
                "cache_key": cache_key,
                "evidence_count": len(evidence)
            }
        }
        
        # Cache result
        if use_cache:
            self._save_to_cache(cache_key, result)
            
        print("✅ Session brief generated successfully!")
        return result

def main():
    """Example usage"""
    # Initialize brief generator
    brief_gen = RAGSessionBrief()
    
    # Example 1: Work stress
    print("\n" + "="*60)
    print("Example 1: Work Stress Scenario")
    print("="*60)
    
    result1 = brief_gen.generate_brief(
        situation="업무로 인한 스트레스가 심하고 집중이 어려운 상황",
        emotions=["stress", "anxiety", "overwhelmed"]
    )
    
    print("\n📋 Generated Brief:")
    print(json.dumps(result1["brief"], ensure_ascii=False, indent=2))
    
    # Example 2: Evening relaxation
    print("\n" + "="*60)
    print("Example 2: Evening Relaxation Scenario") 
    print("="*60)
    
    result2 = brief_gen.generate_brief(
        situation="하루 일과를 마치고 집에서 휴식을 취하고 싶은 저녁 시간",
        emotions=["tired", "peaceful", "contemplative"]
    )
    
    print("\n📋 Generated Brief:")
    print(json.dumps(result2["brief"], ensure_ascii=False, indent=2))
    
    print(f"\n💾 Results cached in: {brief_gen.cache_dir}")

if __name__ == "__main__":
    main()