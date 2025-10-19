import os
import uuid
from datetime import datetime
from typing import List, Optional, Literal, TypedDict
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.store.base import BaseStore

# Pydantic for schemas
from pydantic import BaseModel, Field

# Trustcall for memory management
from trustcall import create_extractor

# OpenAI
from langchain_openai import ChatOpenAI

# Async and threading
import asyncio
import concurrent.futures
from functools import partial

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1",
    temperature=0.7
)

# =============================================================================
# STATE SCHEMA
# =============================================================================

class EmotionalArtState(MessagesState):
    """Emotional art state for extended conversation tracking"""
    
    # Core conversation data
    current_phase: str = "greeting"
    
    # Extended conversation tracking
    conversation_depth: int = 0
    min_sensing_turns: int = 4
    
    # Gradual emotion/situation building
    emotion_hints: List[str] = []
    situation_hints: List[str] = []
    
    # Final detected state
    final_situation: str = ""
    final_emotions: List[str] = []
    confidence_score: float = 0.0
    
    # Recommendation flow control
    consent_for_reco: bool = False
    last_recommendations: List[dict] = []
    
    # Memory and safety
    memory_enabled: bool = False
    safety_notes: str = ""

# =============================================================================
# MEMORY SCHEMAS
# =============================================================================

class UserProfile(BaseModel):
    """User profile schema"""
    name: Optional[str] = Field(description="User Name", default=None)
    location: Optional[str] = Field(description="User Location", default=None)
    job: Optional[str] = Field(description="Job", default=None)
    emotional_context: Optional[str] = Field(description="Current Emotional State", default=None)
    life_situation: Optional[str] = Field(description="Current Life Situation", default=None)
    connections: List[str] = Field(description="Relationships", default_factory=list)
    interests: List[str] = Field(description="Interests or Hobbies", default_factory=list)

class ArtPreferences(BaseModel):
    """Art preferences schema"""
    liked_styles: List[str] = Field(description="Liked Art Styles", default_factory=list)
    avoided_topics: List[str] = Field(description="Topics to Avoid", default_factory=list)
    helpful_motifs: List[str] = Field(description="Helpful Motives or Themes", default_factory=list)
    effective_colors: List[str] = Field(description="Effective Colors", default_factory=list)
    calming_themes: List[str] = Field(description="Calming Themes", default_factory=list)
    feedback_history: List[str] = Field(description="Feedback about Artwork", default_factory=list)

class InteractionPatterns(BaseModel):
    """Interaction patterns schema"""
    preferred_communication_style: Optional[str] = Field(description="Preferred Communication Style", default=None)
    effective_approaches: List[str] = Field(description="Effective Approaches", default_factory=list)
    conversation_preferences: List[str] = Field(description="Conversation Preferences", default_factory=list)
    successful_interventions: List[str] = Field(description="Successful Interventions", default_factory=list)
    user_feedback: List[str] = Field(description="User Feedback", default_factory=list)

class UpdateMemory(TypedDict):
    """Decision on what memory type to update"""
    update_type: Literal['profile', 'art_preferences', 'interaction_patterns']

# =============================================================================
# CONVERSATION PHASE MANAGER
# =============================================================================

class ConversationPhaseManager:
    """Handle extended conversation flow within the main node"""

    def __init__(self, llm):
        self.llm = llm

    async def detect_phase(self, state: EmotionalArtState) -> str:
        """Determine current conversation phase based on extended conversation tracking - consider natural curation transition"""
        # Check if this is the first message (only one HumanMessage exists)
        from langchain_core.messages import HumanMessage
        human_messages = [msg for msg in state.get("messages", []) if isinstance(msg, HumanMessage)]
        
        current_phase = state.get("current_phase", "")
        consent_for_reco = state.get("consent_for_reco")
        final_situation = state.get("final_situation")
        final_emotions = state.get("final_emotions")
        conversation_depth = state.get("conversation_depth", 0)
        
        print(f"DEBUG detect_phase: current_phase={current_phase}, consent_for_reco={consent_for_reco}, final_situation={bool(final_situation)}, final_emotions={bool(final_emotions)}, conversation_depth={conversation_depth}, human_messages={len(human_messages)}")
        
        if len(human_messages) <= 1:
            print("DEBUG detect_phase: Returning 'greeting'")
            return "greeting"
        elif state.get("conversation_depth", 0) < 3:  # Continue conversation for at least 3 turns
            print("DEBUG detect_phase: Returning 'deep_sensing' (depth < 3)")
            return "deep_sensing"
        elif not state.get("final_situation"):
            # After 3+ turns, check curation readiness
            readiness = await self.assess_curation_readiness(state)
            if readiness["ready"] and readiness["confidence"] >= 0.7:
                print("DEBUG detect_phase: Returning 'final_analysis' (ready for curation)")
                return "final_analysis"
            else:
                print("DEBUG detect_phase: Returning 'deep_sensing' (not ready for curation)")
                return "deep_sensing"  # Still need more conversation
        elif state.get("current_phase") == "post_curation":
            # After showing art recommendations, continue natural conversation - MUST check this first!
            print("DEBUG detect_phase: Returning 'continuing' (post_curation)")
            return "continuing"
        elif state.get("current_phase") == "continuing":
            # After any kind of curation attempt (success or failure), continue natural conversation
            print("DEBUG detect_phase: Returning 'continuing' (current_phase=continuing)")
            return "continuing"
        elif state.get("consent_for_reco") == True:
            print("DEBUG detect_phase: Returning 'providing_curation'")
            return "providing_curation"
        elif (
            state.get("final_situation")
            and state.get("final_emotions")
            and not state.get("consent_for_reco")
        ):
            print("DEBUG detect_phase: Returning 'offering'")
            return "offering"
        elif state.get("consent_for_reco") == False:
            # Continue conversation when user declines curation
            print("DEBUG detect_phase: Returning 'continuing_after_rejection'")
            return "continuing_after_rejection"
        else:
            print("DEBUG detect_phase: Returning default 'continuing'")
            return "continuing"

    async def extract_emotion_hints(self, message_content: str) -> List[str]:
        """Extract emotion hints from message using LLM"""
        emotion_prompt = f"""
        Analyze this message and extract emotional indicators:
        
        Message: "{message_content}"
        
        Extract 2-4 key emotional states or feelings expressed or implied.
        Return only the emotions as a comma-separated list (e.g., "stressed, anxious, overwhelmed").
        If no clear emotions, return "neutral".
        """

        try:
            response = await self.llm.ainvoke([SystemMessage(content=emotion_prompt)])
            emotions_str = response.content.strip()

            if emotions_str.lower() == "neutral":
                return []

            emotions = [e.strip() for e in emotions_str.split(',')]
            return emotions[:4]
        except Exception as e:
            print(f"Emotion extraction error: {e}")
            return []

    async def extract_situation_hints(self, message_content: str) -> List[str]:
        """Extract situation hints from message using LLM"""
        situation_prompt = f"""
        Analyze this message and extract situational context:
        
        Message: "{message_content}"
        
        Extract 2-4 key situational elements or life contexts mentioned or implied.
        Focus on: work, relationships, health, finances, education, family, etc.
        Return only the situations as a comma-separated list (e.g., "work pressure, relationship conflict").
        If no clear situation, return "general".
        """

        try:
            response = await self.llm.ainvoke([SystemMessage(content=situation_prompt)])
            situations_str = response.content.strip()

            if situations_str.lower() == "general":
                return []

            situations = [s.strip() for s in situations_str.split(',')]
            return situations[:4]
        except Exception as e:
            print(f"Situation extraction error: {e}")
            return []

    async def generate_contextual_question(self, state: EmotionalArtState, user_context: dict = None) -> str:
        """Generate empathetic response and question based on conversation context"""
        recent_messages = state.get("messages", [])[-4:]
        conversation_text = "\n".join(
            [f"{msg.type}: {msg.content}" for msg in recent_messages]
        )
        
        print(f"DEBUG generate_contextual_question: Called with conversation_text: {conversation_text}")
        print(f"DEBUG generate_contextual_question: Current phase: {state.get('current_phase', '')}")

        emotion_hints = state.get("emotion_hints", [])
        situation_hints = state.get("situation_hints", [])
        depth = state.get("conversation_depth", 0)
        
        # Extract user name
        user_name = ""
        if user_context:
            user_name = extract_user_name_from_memory(user_context)
            print(f"DEBUG: Using name '{user_name}' for contextual question")

        name_context = f"User's name: {user_name}" if user_name else "User's name: Not known yet"
        
        # Check if this is post-curation and user asked an art-related question
        current_phase = state.get("current_phase", "")
        is_post_curation = current_phase == "post_curation"
        
        question_prompt = f"""
        You are a warm, friendly companion who genuinely cares about the user. Think of yourself as a close friend 
        who's a great listener and has knowledge about art and emotions. You're having a natural conversation 
        to understand their emotional state and life situation.
        
        {name_context}
        
        Current conversation:
        {conversation_text}
        
        Emotion hints collected so far: {emotion_hints}
        Situation hints collected so far: {situation_hints}
        Conversation depth: {depth}
        Post-curation context: {is_post_curation}
        
        SPECIAL INSTRUCTION: If the user just asked about art-related topics (like drawing, painting techniques, art tips, 
        art history, etc.) especially after receiving art recommendations, respond directly to their art question with 
        helpful, knowledgeable advice. Be an enthusiastic art companion who can share practical tips and encouragement.
        
        Generate a response that feels like talking to a close friend who really gets you.
        
        Guidelines:
        - If user asked an art-related question, answer it directly and enthusiastically with practical advice
        - Start with genuine empathy and understanding (like a friend would)
        - Use casual, warm language - avoid formal or clinical tone  
        - IMPORTANT: If the user's name is provided above, use EXACTLY that name (never make up or use different names)
        - Use the user's name naturally and sparingly (maybe 1 in 4-5 responses, not every time)
        - Show that you really care about what they're going through with varied, natural expressions
        - Vary your empathetic responses naturally - don't always use the same phrases
        - Be genuinely responsive to what they just said
        - Ask follow-up questions like a caring friend would
        - Be conversational and relatable, not formal or repetitive
        
        Structure your response as:
        1. Direct answer to their question (if art-related) OR friend-like empathetic validation
        2. Natural follow-up question (conversational, not clinical)
        
        IMPORTANT: Don't use the same opening phrases repeatedly. Mix it up naturally:
        - Sometimes start directly with empathy 
        - Sometimes acknowledge what they said first
        - Sometimes use their name, sometimes don't
        - Avoid starting every response with "Oh" or similar exclamations
        
        Generate a warm, friend-like response:
        """

        try:
            response = await self.llm.ainvoke([SystemMessage(content=question_prompt)])
            return response.content.strip()
        except Exception as e:
            print(f"Question generation error: {e}")
            return "Hey, it sounds like you've got a lot on your mind. What's been the biggest thing bothering you lately?"

    async def assess_curation_readiness(self, state: EmotionalArtState) -> dict:
        """Determine natural timing for curation proposal based on conversation context"""
        recent_messages = state.get("messages", [])[-6:]
        conversation_text = "\n".join(
            [f"{msg.type}: {msg.content}" for msg in recent_messages]
        )
        
        emotion_hints = state.get("emotion_hints", [])
        situation_hints = state.get("situation_hints", [])
        depth = state.get("conversation_depth", 0)
        
        readiness_prompt = f"""
        Analyze this conversation to determine if it's a natural time to offer art curation.
        
        Conversation:
        {conversation_text}
        
        Collected emotion hints: {emotion_hints}
        Collected situation hints: {situation_hints}
        Conversation depth: {depth}
        
        Consider these factors:
        1. COMPLETENESS: Has the user shared enough about their situation and feelings?
        2. EMOTIONAL_OPENNESS: Does the user seem open to receiving help or support?
        3. NATURAL_PAUSE: Is there a natural pause or conclusion in their sharing?
        4. READINESS_SIGNALS: Any hints they want help, advice, or something to make them feel better?
        5. CONVERSATION_FLOW: Would offering art recommendations feel natural and helpful right now?
        
        Minimum requirements:
        - At least 3 conversational turns
        - Clear emotional context established
        - User seems to have shared their main concerns
        
        Respond in this exact format:
        READY: [yes/no]
        CONFIDENCE: [0.0-1.0]
        NATURAL_TRANSITION: [suggested transition phrase to use]
        REASON: [brief explanation of why ready or not ready]
        """
        
        try:
            response = await self.llm.ainvoke([SystemMessage(content=readiness_prompt)])
            content = response.content.strip()
            
            lines = content.split('\n')
            ready = False
            confidence = 0.0
            transition = "I wonder if some artwork might help you process these feelings?"
            reason = ""
            
            for line in lines:
                if line.startswith('READY:'):
                    ready = line.replace('READY:', '').strip().lower() == "yes"
                elif line.startswith('CONFIDENCE:'):
                    confidence = float(line.replace('CONFIDENCE:', '').strip())
                elif line.startswith('NATURAL_TRANSITION:'):
                    transition = line.replace('NATURAL_TRANSITION:', '').strip()
                elif line.startswith('REASON:'):
                    reason = line.replace('REASON:', '').strip()
            
            return {
                "ready": ready,
                "confidence": confidence,
                "natural_transition": transition,
                "reason": reason
            }
            
        except Exception as e:
            print(f"Readiness assessment error: {e}")
            return {
                "ready": False,
                "confidence": 0.0,
                "natural_transition": "I wonder if some artwork might help you process these feelings?",
                "reason": "Assessment failed"
            }

    async def analyze_user_consent(self, user_message: str) -> dict:
        """Analyze user's intent for curation consent/rejection"""
        consent_prompt = f"""
        Analyze this user response to determine if they want art recommendations or prefer to continue talking.
        
        User message: "{user_message}"
        
        Look for:
        - ACCEPTANCE: "yes", "sure", "I'd like that", "sounds good", "please show me"
        - REJECTION: "no", "not now", "I want to talk more", "let's continue chatting", "maybe later"
        - NEW_TOPIC: User starts talking about a completely new topic (indicates they want to continue conversation)
        - UNCLEAR: ambiguous responses that need clarification
        
        If the user introduces a NEW TOPIC or shares new personal information, treat this as REJECTION of art curation.
        
        Respond in this exact format:
        CONSENT: [yes/no/unclear]
        CONFIDENCE: [0.0-1.0]
        REASON: [brief explanation of the decision]
        """
        
        try:
            response = await self.llm.ainvoke([SystemMessage(content=consent_prompt)])
            content = response.content.strip()
            
            lines = content.split('\n')
            consent = "unclear"
            confidence = 0.0
            reason = ""
            
            for line in lines:
                if line.startswith('CONSENT:'):
                    consent = line.replace('CONSENT:', '').strip().lower()
                elif line.startswith('CONFIDENCE:'):
                    confidence = float(line.replace('CONFIDENCE:', '').strip())
                elif line.startswith('REASON:'):
                    reason = line.replace('REASON:', '').strip()
            
            return {
                "consent": consent,
                "confidence": confidence,
                "reason": reason
            }
            
        except Exception as e:
            print(f"Consent analysis error: {e}")
            return {
                "consent": "unclear",
                "confidence": 0.0,
                "reason": "Analysis failed"
            }

    async def check_art_question(self, message_content: str) -> dict:
        """Check if user message is asking an art-related question (vs simple consent)"""
        if isinstance(message_content, list):
            message_content = " ".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in message_content])
        
        art_check_prompt = f"""
        Analyze this message to determine if it's an art-related question or request:
        
        Message: "{message_content}"
        
        Determine if this is:
        1. A simple consent response (like "yes", "no", "sure", "okay") 
        2. An art-related question or request (asking about drawing, painting, art techniques, art history, creative advice, etc.)
        
        Return in this format:
        IS_ART_QUESTION: [true/false]
        CONFIDENCE: [0.0-1.0]
        REASON: [brief explanation]
        """
        
        try:
            response = await self.llm.ainvoke([SystemMessage(content=art_check_prompt)])
            content = response.content.strip()
            
            is_art_question = False
            confidence = 0.0
            reason = "Analysis failed"
            
            lines = content.split('\n')
            for line in lines:
                if line.startswith('IS_ART_QUESTION:'):
                    is_art_question = line.replace('IS_ART_QUESTION:', '').strip().lower() == "true"
                elif line.startswith('CONFIDENCE:'):
                    confidence = float(line.replace('CONFIDENCE:', '').strip())
                elif line.startswith('REASON:'):
                    reason = line.replace('REASON:', '').strip()
            
            return {
                "is_art_question": is_art_question,
                "confidence": confidence,
                "reason": reason
            }
            
        except Exception as e:
            print(f"Art question check error: {e}")
            return {
                "is_art_question": False,
                "confidence": 0.0,
                "reason": "Analysis failed"
            }

    async def perform_final_analysis(self, state: EmotionalArtState) -> dict:
        """Analyze all collected hints and make final emotion/situation determination"""
        all_emotion_hints = state.get("emotion_hints", [])
        all_situation_hints = state.get("situation_hints", [])
        recent_messages = state.get("messages", [])[-8:]

        conversation_text = "\n".join(
            [f"{msg.type}: {msg.content}" for msg in recent_messages]
        )

        analysis_prompt = f"""
        Based on this conversation and collected hints, provide a comprehensive analysis:
        
        Conversation:
        {conversation_text}
        
        Collected emotion hints: {all_emotion_hints}
        Collected situation hints: {all_situation_hints}
        
        Please analyze and provide:
        1. A concise situation description (1-2 sentences, describing the main life context/challenge)
        2. Primary emotions (3-5 key emotional states)
        3. Confidence score (0.0-1.0, how confident you are in this analysis)
        
        Respond in this exact format:
        SITUATION: [detailed situation description]
        EMOTIONS: [emotion1, emotion2, emotion3, emotion4, emotion5]
        CONFIDENCE: [0.0-1.0]
        """

        try:
            response = await self.llm.ainvoke([SystemMessage(content=analysis_prompt)])
            content = response.content

            lines = content.strip().split('\n')
            situation = ""
            emotions = []
            confidence = 0.0

            for line in lines:
                if line.startswith('SITUATION:'):
                    situation = line.replace('SITUATION:', '').strip()
                elif line.startswith('EMOTIONS:'):
                    emotions_str = line.replace('EMOTIONS:', '').strip()
                    emotions = [e.strip() for e in emotions_str.split(',')]
                elif line.startswith('CONFIDENCE:'):
                    confidence = float(line.replace('CONFIDENCE:', '').strip())

            return {
                "situation": situation,
                "emotions": emotions,
                "confidence": confidence
            }
        except Exception as e:
            print(f"Analysis error: {e}")
            return {
                "situation": "general life situation",
                "emotions": ["mixed feelings"],
                "confidence": 0.5
            }

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_user_name_from_memory(user_context: dict) -> str:
    """Extract user name from memory"""
    if user_context.get("profile") and isinstance(user_context["profile"], dict):
        return user_context["profile"].get("name", "")
    return ""

async def load_user_memories(config: RunnableConfig, store: BaseStore) -> dict:
    """Load user memories"""
    # In LangGraph Studio, user_id may not be automatically set
    user_id = config.get("configurable", {}).get("user_id") or config.get("configurable", {}).get("thread_id", "default_user")

    # Load profile memory using fixed key
    profile_namespace = ("profile", user_id)
    try:
        profile_mem = await store.aget(profile_namespace, "user_profile")
        profile = profile_mem.value if profile_mem else None
    except:
        # Fallback to search if direct get fails
        profile_memories = await store.asearch(profile_namespace)
        profile = profile_memories[0].value if profile_memories else None

    # Load art preferences memory
    art_namespace = ("art_preferences", user_id)
    art_memories = await store.asearch(art_namespace)
    art_preferences = [mem.value for mem in art_memories]

    # Load interaction patterns memory
    interaction_namespace = ("interaction_patterns", user_id)
    interaction_memories = await store.asearch(interaction_namespace)
    interaction_patterns = [mem.value for mem in interaction_memories]

    return {
        "profile": profile,
        "art_preferences": art_preferences,
        "interaction_patterns": interaction_patterns
    }

def generate_greeting_response(user_context: dict) -> str:
    """Generate greeting response"""
    return "Hey there! I'm here to chat and maybe help you discover some amazing art along the way. How's your day going?"

def generate_continuing_response(state: EmotionalArtState, user_context: dict) -> str:
    """Generate continuing conversation response"""
    return "How are you feeling about the artwork? Would you like to explore more, or is there something else on your mind?"

async def analyze_for_memory_updates(state: EmotionalArtState, response: str, user_context: dict, llm) -> dict:
    """Analyze memory update necessity using LLM"""
    recent_messages = state["messages"][-4:]
    conversation_text = "\n".join([
        f"{msg.type}: {msg.content}" for msg in recent_messages
    ])

    current_profile = user_context.get("profile", "No profile stored")
    current_art_prefs = user_context.get("art_preferences", "No art preferences stored")
    current_interactions = user_context.get("interaction_patterns", "No interaction patterns stored")
    
    # Add emotion_hints and situation_hints information
    emotion_hints = state.get("emotion_hints", [])
    situation_hints = state.get("situation_hints", [])

    analysis_prompt = f"""
    Analyze this conversation to determine if any memory should be updated:
    
    Recent conversation:
    {conversation_text}
    
    Collected conversation insights:
    - Emotion hints: {emotion_hints}
    - Situation hints: {situation_hints}
    
    Current memory state:
    - Profile: {current_profile}
    - Art preferences: {current_art_prefs}
    - Interaction patterns: {current_interactions}
    
    IMPORTANT: Only recommend updates if the user has EXPLICITLY shared NEW information that is not already stored.
    Consider emotion_hints and situation_hints as valid information collected from natural conversation.
    
    Determine if any of these should be updated based on the conversation:
    1. PROFILE: Personal information (name, location, job, relationships, interests) - only if user explicitly shared these details
    2. ART_PREFERENCES: Art styles, colors, themes they like/dislike, effective artwork - only if user mentioned specific preferences
    3. INTERACTION_PATTERNS: Communication preferences, effective conversation approaches - only if clear patterns emerged
    
    Do NOT recommend updates for:
    - Assumptions or inferences not explicitly stated
    - Information already captured in existing memory
    
    DO recommend updates for:
    - When emotion_hints have been collected and can provide insight into emotional context
    - When situation_hints have been collected and can provide insight into life situation
    - When user shares new personal information not in existing memory
    - When enough conversation context has been gathered to update emotional_context or life_situation
    
    Respond in this exact format:
    UPDATE_NEEDED: [yes/no]
    UPDATE_TYPE: [profile/art_preferences/interaction_patterns/none]
    REASON: [brief explanation why update is needed or not needed]
    """

    try:
        llm_response = await llm.ainvoke([SystemMessage(content=analysis_prompt)])
        content = llm_response.content.strip()

        lines = content.split('\n')
        update_needed = False
        update_type = "none"
        reason = ""

        for line in lines:
            line = line.strip()
            if line.startswith('UPDATE_NEEDED:'):
                value = line.replace('UPDATE_NEEDED:', '').strip().lower()
                update_needed = value == "yes" or value == "true"
            elif line.startswith('UPDATE_TYPE:'):
                update_type = line.replace('UPDATE_TYPE:', '').strip()
                # Remove brackets if present
                update_type = update_type.strip('[]')
            elif line.startswith('REASON:'):
                reason = line.replace('REASON:', '').strip()
                # Remove brackets if present
                reason = reason.strip('[]')

        # If update_type is anything other than "none", set update_needed to True
        if update_type in ["profile", "art_preferences", "interaction_patterns"]:
            update_needed = True

        print(f"DEBUG: Parsed memory analysis - needed: {update_needed}, type: {update_type}, reason: {reason}")
        
        return {
            "update_needed": update_needed,
            "update_type": update_type,
            "reason": reason
        }

    except Exception as e:
        print(f"Memory analysis error: {e}")
        return {
            "update_needed": False,
            "update_type": "none",
            "reason": "Analysis failed"
        }

# =============================================================================
# THREAD-SAFE MEMORY PROCESSOR
# =============================================================================

class ThreadSafeMemoryProcessor:
    """Handles memory updates in separate thread pool to avoid async/sync conflicts"""
    
    def __init__(self, max_workers=2):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
    def shutdown(self):
        """Shutdown the thread pool"""
        self.executor.shutdown(wait=True)
    
    async def process_memory_update(self, memory_type: str, messages: list, 
                                  existing_memories: list, namespace: tuple, 
                                  store: BaseStore, emotion_hints: list = None, 
                                  situation_hints: list = None):
        """Process memory update in separate thread"""
        
        # Create partial function with all synchronous operations
        sync_update_func = partial(
            self._sync_memory_update,
            memory_type=memory_type,
            messages=messages,
            existing_memories=existing_memories,
            namespace=namespace,
            emotion_hints=emotion_hints,
            situation_hints=situation_hints
        )
        
        try:
            # Run in thread pool to avoid event loop conflicts
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor, 
                sync_update_func
            )
            
            # Store results back to LangGraph store (async)
            if result and result.get('updates'):
                for update_data in result['updates']:
                    await store.aput(
                        namespace, 
                        update_data['key'], 
                        update_data['value']
                    )
            
            return result
            
        except Exception as e:
            print(f"DEBUG: Memory processing failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _sync_memory_update(self, memory_type: str, messages: list, 
                           existing_memories: list, namespace: tuple,
                           emotion_hints: list = None, situation_hints: list = None):
        """Synchronous memory update - runs in thread pool"""
        
        # Import here to avoid circular imports in thread
        from langchain_core.messages import SystemMessage
        
        try:
            if memory_type == "profile":
                return self._update_profile_sync(messages, existing_memories, emotion_hints, situation_hints)
            elif memory_type == "art_preferences":
                return self._update_art_preferences_sync(messages, existing_memories)
            elif memory_type == "interaction_patterns":
                return self._update_interaction_patterns_sync(messages, existing_memories)
            else:
                return None
                
        except Exception as e:
            print(f"DEBUG: Sync memory update failed: {e}")
            return None
    
    def _update_profile_sync(self, messages: list, existing_memories: list,
                           emotion_hints: list, situation_hints: list):
        """Synchronous profile update"""
        from langchain_core.messages import SystemMessage

        # Create instruction with context
        context_info = ""
        if emotion_hints:
            context_info += f"\nCurrent emotion patterns: {', '.join(emotion_hints[-5:])}"
        if situation_hints:
            context_info += f"\nCurrent situation context: {', '.join(situation_hints[-5:])}"

        trustcall_instruction = f"""
        Update user profile based on this conversation.
        Extract personal information: name, location, job, emotional context, life situation,
        relationships, interests/hobbies.{context_info}
        System Time: {datetime.now().isoformat()}
        """

        messages_for_extraction = [SystemMessage(content=trustcall_instruction)] + messages

        class SyncSpy:
            def __init__(self):
                self.called_tools = []
            def __call__(self, run):
                pass  # Simplified for thread execution

        spy = SyncSpy()
        profile_extractor_with_spy = profile_extractor.with_listeners(on_end=spy)

        result = profile_extractor_with_spy.invoke({
            "messages": messages_for_extraction,
            "existing": existing_memories
        })

        # Use fixed key for single profile per user
        updates = []
        for r in result["responses"]:
            updates.append({
                'key': "user_profile",  # Fixed key - always update the same profile
                'value': r.model_dump(mode="json")
            })

        return {'updates': updates, 'type': 'profile'}
    
    def _update_art_preferences_sync(self, messages: list, existing_memories: list):
        """Synchronous art preferences update"""
        from langchain_core.messages import SystemMessage
        
        trustcall_instruction = f"""
        Update art preferences based on this conversation.
        Extract liked/disliked art styles, effective colors, helpful themes, user feedback on artworks.
        System Time: {datetime.now().isoformat()}
        """
        
        messages_for_extraction = [SystemMessage(content=trustcall_instruction)] + messages[:-1]
        
        class SyncSpy:
            def __init__(self):
                self.called_tools = []
            def __call__(self, run):
                pass
        
        spy = SyncSpy()
        art_extractor_with_spy = art_preference_extractor.with_listeners(on_end=spy)
        
        result = art_extractor_with_spy.invoke({
            "messages": messages_for_extraction,
            "existing": existing_memories
        })
        
        updates = []
        for r, rmeta in zip(result["responses"], result["response_metadata"]):
            updates.append({
                'key': rmeta.get("json_doc_id", str(uuid.uuid4())),
                'value': r.model_dump(mode="json")
            })
        
        return {'updates': updates, 'type': 'art_preferences'}
    
    def _update_interaction_patterns_sync(self, messages: list, existing_memories: list):
        """Synchronous interaction patterns update"""
        from langchain_core.messages import SystemMessage
        
        trustcall_instruction = f"""
        Update interaction patterns based on this conversation.
        Extract communication preferences, effective approaches, successful conversation strategies, 
        user feedback.
        System Time: {datetime.now().isoformat()}
        """
        
        messages_for_extraction = [SystemMessage(content=trustcall_instruction)] + messages[:-1]
        
        class SyncSpy:
            def __init__(self):
                self.called_tools = []
            def __call__(self, run):
                pass
        
        spy = SyncSpy()
        interaction_extractor_with_spy = interaction_extractor.with_listeners(on_end=spy)
        
        result = interaction_extractor_with_spy.invoke({
            "messages": messages_for_extraction,
            "existing": existing_memories
        })
        
        updates = []
        for r, rmeta in zip(result["responses"], result["response_metadata"]):
            updates.append({
                'key': rmeta.get("json_doc_id", str(uuid.uuid4())),
                'value': r.model_dump(mode="json")
            })
        
        return {'updates': updates, 'type': 'interaction_patterns'}

# Global memory processor instance
memory_processor = ThreadSafeMemoryProcessor()

# =============================================================================
# TRUSTCALL EXTRACTORS
# =============================================================================

class Spy:
    def __init__(self):
        self.called_tools = []
        
    def __call__(self, run):
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )

# Create extractors
profile_extractor = create_extractor(
    llm,
    tools=[UserProfile],
    tool_choice="UserProfile",
)

art_preference_extractor = create_extractor(
    llm,
    tools=[ArtPreferences],
    tool_choice="ArtPreferences",
    enable_inserts=True
)

interaction_extractor = create_extractor(
    llm,
    tools=[InteractionPatterns],
    tool_choice="InteractionPatterns",
    enable_inserts=True
)

# =============================================================================
# NODES
# =============================================================================

async def conversation_node(state: EmotionalArtState, config: RunnableConfig, store: BaseStore):
    """Conversation-only node - emotion extraction, memory updates, basic response handling"""
    
    print("=== NEW CONVERSATION_NODE STARTED ===")
    print(f"State: {list(state.keys())}")
    print("=== NEW CONVERSATION_NODE STARTED ===")
    
    # Add store validation and fallback handling
    if store is None:
        print("WARNING: Store is None, using InMemoryStore fallback")
        from langgraph.store.memory import InMemoryStore
        store = InMemoryStore()
    
    try:
        user_id = config.get("configurable", {}).get("user_id") or config.get("configurable", {}).get("thread_id", "default_user")
        user_context = await load_user_memories(config, store)
        
        phase_manager = ConversationPhaseManager(llm)
        
        current_phase = await phase_manager.detect_phase(state)
        
        # Get the last user message
        last_user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_user_message = msg.content
                break
        
        print(f"DEBUG: Phase={current_phase}, User message={last_user_message}")
        
        # Prepare dictionary for state updates
        state_updates = {}
        
        # Extract hints from all user messages (always run for stability)
        if last_user_message:
            try:
                emotion_hints = await phase_manager.extract_emotion_hints(last_user_message)
                situation_hints = await phase_manager.extract_situation_hints(last_user_message)
                
                print(f"DEBUG: Raw extracted hints - emotions: {emotion_hints}, situations: {situation_hints}")
                
                # Merge with existing hints, remove duplicates, and keep max 5
                current_emotion_hints = state.get("emotion_hints", [])
                current_situation_hints = state.get("situation_hints", [])
                
                # Add new hints and remove duplicates (maintain order)
                all_emotion_hints = current_emotion_hints + emotion_hints
                all_situation_hints = current_situation_hints + situation_hints
                
                # Remove duplicates while maintaining order, limit to max 5
                state_updates["emotion_hints"] = list(dict.fromkeys(all_emotion_hints))[-5:]
                state_updates["situation_hints"] = list(dict.fromkeys(all_situation_hints))[-5:]
                
                print(f"DEBUG: Final merged hints - emotions: {state_updates['emotion_hints']}, situations: {state_updates['situation_hints']}")
                
            except Exception as e:
                print(f"DEBUG: Error extracting hints: {e}")
                # Maintain existing hints on error
                state_updates["emotion_hints"] = state.get("emotion_hints", [])
                state_updates["situation_hints"] = state.get("situation_hints", [])
        else:
            # If no message, maintain existing hints
            state_updates["emotion_hints"] = state.get("emotion_hints", [])
            state_updates["situation_hints"] = state.get("situation_hints", [])
        
        if current_phase == "greeting":
            response_content = generate_greeting_response(user_context)
        
        elif current_phase == "deep_sensing":
            # conversation_depth is incremented only in deep_sensing
            state_updates["conversation_depth"] = state.get("conversation_depth", 0) + 1
            response_content = await phase_manager.generate_contextual_question(state, user_context)
        
        elif current_phase == "final_analysis":
            # First re-check readiness to get natural transition phrase
            readiness = await phase_manager.assess_curation_readiness(state)
            analysis_result = await phase_manager.perform_final_analysis(state)
            
            state_updates["final_situation"] = analysis_result["situation"]
            state_updates["final_emotions"] = analysis_result["emotions"]
            state_updates["confidence_score"] = analysis_result["confidence"]
            
            if analysis_result["confidence"] >= 0.7:
                state_updates["current_phase"] = "offering"
                # Use natural transition phrase
                natural_transition = readiness.get("natural_transition", "You know what, I'm thinking some artwork might really help with what you're going through...")
                response_content = f"{natural_transition} Want me to find some pieces that might really speak to you right now?"
            else:
                state_updates["current_phase"] = "deep_sensing"
                response_content = await phase_manager.generate_contextual_question(state, user_context)
        
        elif current_phase == "offering":
            # Analyze user consent/rejection
            if last_user_message:
                consent_analysis = await phase_manager.analyze_user_consent(last_user_message)
                
                print(f"DEBUG: Consent analysis - {consent_analysis}")
                
                if consent_analysis["consent"] == "yes" and consent_analysis["confidence"] >= 0.7:
                    state_updates["consent_for_reco"] = True
                    response_content = "Awesome! Let me find some pieces that might really speak to you right now."
                elif consent_analysis["consent"] == "no" and consent_analysis["confidence"] >= 0.7:
                    state_updates["consent_for_reco"] = False
                    response_content = "Totally get it! I'm here to chat whenever you need. What else is going on with you?"
                else:
                    # Ask clearly when uncertain
                    response_content = "I'm happy either way - want me to show you some cool art pieces, or would you rather keep talking about what's on your mind?"
            else:
                response_content = "What do you think - want me to find some cool artwork that might vibe with how you're feeling, or would you rather keep chatting?"
        
        elif current_phase == "providing_curation":
            # Curation phase handled by separate node, only update phase
            state_updates["current_phase"] = "ready_for_curation"
            response_content = "Let me find some perfect pieces for you..."
            
        elif current_phase == "continuing_after_rejection":
            # Continue conversation after rejecting curation - hint extraction already handled above
            state_updates["conversation_depth"] = state.get("conversation_depth", 0) + 1
            
            if last_user_message:
                # Respond empathetically to new topic
                response_content = await phase_manager.generate_contextual_question(state, user_context)
            else:
                response_content = "What's been on your mind lately?"
        
        elif current_phase == "continuing":
            # Use intelligent contextual response instead of generic post-curation response
            print(f"DEBUG: In continuing phase, last_user_message: {last_user_message}")
            if last_user_message:
                print("DEBUG: Calling generate_contextual_question for post-curation response")
                response_content = await phase_manager.generate_contextual_question(state, user_context)
            else:
                print("DEBUG: No last_user_message, using generic continuing response")
                response_content = generate_continuing_response(state, user_context)
        
        else:
            response_content = "Hey, I'm here for you. What's going on?"
        
        # Decide memory update - always update profile if hints exist
        current_emotion_hints = state_updates.get("emotion_hints", [])
        current_situation_hints = state_updates.get("situation_hints", [])
        
        # Update profile if hints are newly added or existing hints exist
        if current_emotion_hints or current_situation_hints:
            memory_analysis = {
                "update_needed": True,
                "update_type": "profile",
                "reason": f"Emotion hints ({current_emotion_hints}) or situation hints ({current_situation_hints}) collected"
            }
            print(f"DEBUG: Auto-triggered profile update due to hints")
        else:
            memory_analysis = await analyze_for_memory_updates(state, response_content, user_context, llm)
        
        print(f"DEBUG: Memory analysis={memory_analysis}")
        print(f"DEBUG: Response={response_content[:50]}...")
        print(f"DEBUG: Current emotion hints in state: {state.get('emotion_hints', [])}")
        print(f"DEBUG: Current situation hints in state: {state.get('situation_hints', [])}")
        
        # state_updates already includes emotion_hints and situation_hints, remove duplicate processing
        # Already extracted by phase above, so only maintain existing values here
        if "emotion_hints" not in state_updates:
            state_updates["emotion_hints"] = state.get("emotion_hints", [])
        if "situation_hints" not in state_updates:
            state_updates["situation_hints"] = state.get("situation_hints", [])
            
        print(f"DEBUG: Final emotion hints: {state_updates['emotion_hints']}")
        print(f"DEBUG: Final situation hints: {state_updates['situation_hints']}")
        print(f"DEBUG: All state updates: {list(state_updates.keys())}")

        # Process memory update in background if needed
        if memory_analysis["update_needed"]:
            # Validate if update_type is valid (enhance stability)
            valid_update_types = ["profile", "art_preferences", "interaction_patterns"]
            update_type = memory_analysis.get("update_type", "")
            
            if update_type not in valid_update_types:
                print(f"DEBUG: Invalid update_type '{update_type}', skipping memory update")
                print(f"DEBUG: Valid types are: {valid_update_types}")
            else:
                # Perform memory update in background
                try:
                    print(f"DEBUG: About to update {update_type} memory")
                    print(f"DEBUG: Recent messages for extraction: {[msg.content for msg in state['messages'][-6:]]}")
                    
                    # Use the new thread-safe memory processor
                    user_id = config.get("configurable", {}).get("user_id") or config.get("configurable", {}).get("thread_id", "default_user")
                    result = None  # Initialize result variable
                    
                    if update_type == "profile":
                        namespace = ("profile", user_id)
                        recent_messages = state["messages"][-3:]
                        user_messages_only = [msg for msg in recent_messages if isinstance(msg, HumanMessage)]
                        
                        existing_items = await store.asearch(namespace)
                        existing_memories = ([(existing_item.key, "UserProfile", existing_item.value) 
                                            for existing_item in existing_items] if existing_items else None)
                        
                        current_emotion_hints = state.get("emotion_hints", [])
                        current_situation_hints = state.get("situation_hints", [])
                        
                        result = await memory_processor.process_memory_update(
                            "profile", user_messages_only, existing_memories, namespace, store,
                            current_emotion_hints, current_situation_hints
                        )
                        
                    elif update_type == "art_preferences":
                        namespace = ("art_preferences", user_id)
                        recent_messages = state["messages"][-6:]
                        
                        existing_items = await store.asearch(namespace)
                        existing_memories = ([(existing_item.key, "ArtPreferences", existing_item.value)
                                            for existing_item in existing_items] if existing_items else None)
                        
                        result = await memory_processor.process_memory_update(
                            "art_preferences", recent_messages, existing_memories, namespace, store
                        )
                        
                    elif update_type == "interaction_patterns":
                        namespace = ("interaction_patterns", user_id)
                        recent_messages = state["messages"][-8:]
                        
                        existing_items = await store.asearch(namespace)
                        existing_memories = ([(existing_item.key, "InteractionPatterns", existing_item.value)
                                            for existing_item in existing_items] if existing_items else None)
                        
                        current_emotion_hints = state_updates.get("emotion_hints", [])
                        current_situation_hints = state_updates.get("situation_hints", [])
                        
                        result = await memory_processor.process_memory_update(
                            "interaction_patterns", recent_messages, existing_memories, namespace, store,
                            current_emotion_hints, current_situation_hints
                        )
                
                    if result:
                        print(f"DEBUG: Memory updated successfully: {update_type} - {len(result.get('updates', []))} items")
                    else:
                        print(f"DEBUG: Memory update returned no result: {update_type}")
                        
                except Exception as e:
                    print(f"DEBUG: Memory update failed: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Always return normal AI response
        return {"messages": [AIMessage(content=response_content)], **state_updates}
    
    except Exception as e:
        print(f"Error in emotional_art_docent: {e}")
        import traceback
        traceback.print_exc()
        return {"messages": [AIMessage(content="I'm here to help. How are you feeling today?")]}

async def update_user_profile(state: EmotionalArtState, config: RunnableConfig, store: BaseStore):
    """Update user profile"""
    
    user_id = config.get("configurable", {}).get("user_id") or config.get("configurable", {}).get("thread_id", "default_user")
    namespace = ("profile", user_id)
    
    existing_items = await store.asearch(namespace)
    existing_memories = ([(existing_item.key, "UserProfile", existing_item.value) 
                         for existing_item in existing_items] if existing_items else None)
    
    # Use only recent messages and filter for actual user messages only
    recent_messages = state["messages"][-3:]  # Use fewer messages
    user_messages_only = [msg for msg in recent_messages if isinstance(msg, HumanMessage)]
    
    print(f"DEBUG: Messages being sent to Trustcall: {[msg.content for msg in user_messages_only]}")
    print(f"DEBUG: Existing memories: {existing_memories}")
    
    # Provide emotion_hints and situation_hints to Trustcall
    current_emotion_hints = state.get("emotion_hints", [])
    current_situation_hints = state.get("situation_hints", [])
    
    trustcall_instruction = f"""
    CRITICAL RULES - FOLLOW EXACTLY:
    
    Available context from conversation analysis:
    - Emotion hints collected: {current_emotion_hints}
    - Situation hints collected: {current_situation_hints}
    
    1. ONLY extract information that appears in EXACT QUOTES from the user's messages
    2. DO NOT infer, assume, deduce, or generate ANY information beyond what's provided
    3. For emotional_context: Use the emotion hints collected from conversation analysis if available
    4. DO NOT create life situations based on context clues
    5. If the user hasn't explicitly stated something with clear words, leave it NULL/empty
    
    EXTRACTION RULES:
    - name: ONLY if user says "My name is X" or "I'm X" or "Call me X"
    - location: ONLY if user says "I live in X" or "I'm from X" 
    - job: ONLY if user says "I work as X" or "My job is X"
    - emotional_context: Based on accumulated emotion patterns from the conversation (use the emotion_hints collected over multiple turns)
    - life_situation: ONLY if user describes a specific current situation with clear details
    - connections: ONLY names explicitly mentioned as relationships
    - interests: ONLY activities explicitly mentioned as hobbies/interests
    
    EXAMPLE - If user only says "My name is Lee":
    - name: Lee
    - location: null (not mentioned)
    - job: null (not mentioned)  
    - emotional_context: null (not mentioned)
    - life_situation: null (not mentioned)
    - connections: [] (not mentioned)
    - interests: [] (not mentioned)
    
    System Time: {datetime.now().isoformat()}
    """
    
    # Use only system instruction and user messages
    messages_for_extraction = [SystemMessage(content=trustcall_instruction)] + user_messages_only
    
    spy = Spy()
    profile_extractor_with_spy = profile_extractor.with_listeners(on_end=spy)
    
    # Pass existing memory to update, not create new
    result = profile_extractor_with_spy.invoke({
        "messages": messages_for_extraction,
        "existing": existing_memories
    })

    # Use fixed key for single profile per user
    for r in result["responses"]:
        await store.aput(namespace,
                         "user_profile",  # Fixed key - always update the same profile
                         r.model_dump(mode="json"))

    # Called from background, no response needed
    return

async def update_art_preferences(state: EmotionalArtState, config: RunnableConfig, store: BaseStore):
    """Update art preferences"""
    
    user_id = config.get("configurable", {}).get("user_id") or config.get("configurable", {}).get("thread_id", "default_user")
    namespace = ("art_preferences", user_id)
    
    existing_items = await store.asearch(namespace)
    existing_memories = ([(existing_item.key, "ArtPreferences", existing_item.value)
                        for existing_item in existing_items]
                        if existing_items else None)
    
    recent_messages = state["messages"][-6:]
    trustcall_instruction = f"""
    Update art preferences based on this conversation.
    Extract liked/disliked art styles, effective colors, helpful themes, user feedback on artworks.
    System Time: {datetime.now().isoformat()}
    """
    
    messages_for_extraction = [SystemMessage(content=trustcall_instruction)] + recent_messages[:-1]
    
    spy = Spy()
    art_extractor_with_spy = art_preference_extractor.with_listeners(on_end=spy)
    
    result = art_extractor_with_spy.invoke({
        "messages": messages_for_extraction,
        "existing": existing_memories
    })

    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        await store.aput(namespace,
                         rmeta.get("json_doc_id", str(uuid.uuid4())),
                         r.model_dump(mode="json"))

    # Called from background, no response needed
    return

async def update_interaction_patterns(state: EmotionalArtState, config: RunnableConfig, store: BaseStore):
    """Update interaction patterns"""

    user_id = config.get("configurable", {}).get("user_id") or config.get("configurable", {}).get("thread_id", "default_user")
    namespace = ("interaction_patterns", user_id)

    existing_items = await store.asearch(namespace)
    existing_memories = ([(existing_item.key, "InteractionPatterns", existing_item.value)
                        for existing_item in existing_items]
                        if existing_items else None)

    recent_messages = state["messages"][-8:]
    trustcall_instruction = f"""
    Update interaction patterns based on this conversation.
    Extract communication preferences, effective approaches, successful conversation strategies, 
    user feedback.
    System Time: {datetime.now().isoformat()}
    """

    messages_for_extraction = [SystemMessage(content=trustcall_instruction)] + recent_messages[:-1]

    spy = Spy()
    interaction_extractor_with_spy = interaction_extractor.with_listeners(on_end=spy)

    result = interaction_extractor_with_spy.invoke({
        "messages": messages_for_extraction,
        "existing": existing_memories
    })

    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        await store.aput(namespace,
                         rmeta.get("json_doc_id", str(uuid.uuid4())),
                         r.model_dump(mode="json"))

    # Called from background, no response needed
    return

# =============================================================================
# ART CURATION ENGINE INTEGRATION (Phase 2)
# =============================================================================

def prepare_curation_input(user_memories: dict, conversation_state: EmotionalArtState) -> dict:
    """Convert user memories + conversation context into art_curation_engine input"""
    
    # 1. Situation aggregation (70% current + 30% background context)
    current_situation = user_memories.get('user_profile', {}).get('current_situation', '')
    emotional_context = user_memories.get('user_profile', {}).get('emotional_context', '')
    
    situation = f"{current_situation}"
    if emotional_context and emotional_context != current_situation:
        situation += f" with underlying {emotional_context}"
    
    # 2. Emotions prioritization (conversation > stored)
    emotions = []
    # Handle both dict and EmotionalArtState access patterns
    print(f"DEBUG: conversation_state type: {type(conversation_state)}")
    print(f"DEBUG: conversation_state content: {conversation_state}")
    
    try:
        if hasattr(conversation_state, 'emotion_hints'):
            emotion_hints = conversation_state.emotion_hints or []
        elif hasattr(conversation_state, 'get'):
            emotion_hints = conversation_state.get('emotion_hints', [])
        else:
            emotion_hints = []
    except Exception as e:
        print(f"DEBUG: Error accessing emotion_hints: {e}")
        emotion_hints = []
    
    if emotion_hints:
        emotions.extend(emotion_hints[:3])  # Top 3 from conversation
    
    stored_emotions = user_memories.get('user_profile', {}).get('emotions', [])
    for emotion in stored_emotions:
        if emotion not in emotions and len(emotions) < 5:
            emotions.append(emotion)
    
    # 3. Art preferences (when available)
    art_prefs = user_memories.get('art_preferences', {})
    preferences = None
    if art_prefs:
        preferences = {
            "liked_styles": art_prefs.get('preferred_styles', [])[:5],
            "avoided_themes": art_prefs.get('avoided_themes', [])[:5],
            "effective_colors": art_prefs.get('effective_colors', [])[:5],
            "preferred_periods": art_prefs.get('preferred_periods', [])[:3]
        }
    
    # 4. Fallbacks
    if not situation.strip():
        situation = "general emotional support and mood enhancement"
    if not emotions:
        emotions = ["neutral", "seeking_inspiration"]
    
    return {
        "situation": situation.strip(),
        "emotions": emotions,
        "preferences": preferences,
        "confidence_level": min(1.0, len(emotions) * 0.2 + (0.3 if preferences else 0))
    }

def format_recommendations_for_chat(final_recs: dict, curation_input: dict, processing_time: float) -> str:
    """Convert engine output to conversational format"""
    
    recommendations = final_recs.get('final_recommendations', [])[:3]  # Top 3 for chat
    confidence = curation_input.get('confidence_level', 0.5)
    has_preferences = bool(curation_input.get('preferences'))
    
    # Confidence-based introduction
    if confidence >= 0.8:
        intro = "Based on what you've shared, I found some artwork that really resonates with your current feelings:"
    elif confidence >= 0.6:
        intro = "I found some artwork that might help with what you're experiencing:"
    else:
        intro = "Here are some pieces that often help people in similar situations:"
    
    response = intro + "\n\n"
    
    # Present top 3 recommendations
    for i, rec in enumerate(recommendations, 1):
        artwork_id = rec.get('artwork_id', 'Unknown')
        score = rec.get('rerank_score', 0)
        justification = rec.get('justification', '')
        
        # Truncate justification for conversation flow
        short_justification = justification[:80] + "..." if len(justification) > 80 else justification
        
        response += f"**{i}. Artwork #{artwork_id}** (Match: {score:.0%})\n"
        response += f"   {short_justification}\n\n"
    
    # Personalization note
    if has_preferences:
        response += "These selections consider your previous art preferences. "
    
    # Call to action
    response += "Would you like to see any of these pieces, or shall I find different options?"
    
    # Processing note (if slow)
    if processing_time > 30:
        response += f"\n\n*Found after {processing_time:.0f}s of research-backed analysis*"
    
    return response

async def handle_curation_request(state: EmotionalArtState, config: RunnableConfig, store: BaseStore):
    """Execute art curation with timeout and error handling"""
    import time
    import asyncio
    import os
    
    # 1. Prepare input
    user_memories = await load_user_memories(config, store)
    print(f"DEBUG: handle_curation_request state type: {type(state)}")
    print(f"DEBUG: handle_curation_request state keys: {list(state.keys()) if hasattr(state, 'keys') else 'no keys method'}")
    curation_input = prepare_curation_input(user_memories, state)
    
    # 2. Change to art_curation_engine working directory (critical for data loading)
    original_cwd = await asyncio.to_thread(os.getcwd)
    # Use the art_curation_engine directory
    art_engine_dir = os.path.join(os.path.dirname(__file__), 'art_curation_engine')
    
    try:
        os.chdir(art_engine_dir)
        print(f"DEBUG: Changed working directory to {art_engine_dir}")
        
        # 3. Import and initialize (lazy loading for performance)
        # Add project root to Python path for imports
        import sys
        project_root = os.path.dirname(art_engine_dir)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from art_curation_engine.core import RAGSessionBrief, StageACollector, Step6LLMReranker
        
        rag_session = RAGSessionBrief()
        stage_a = StageACollector()
        reranker = Step6LLMReranker(
            batch_size=2,  # Reduced batch size
            max_workers=1,  # Single worker to prevent concurrent calls
            api_delay_seconds=3.0,  # 3 second delay between API calls
            max_retries=3  # Retry failed calls up to 3 times
        )
        
    except ImportError as e:
        print(f"Art curation engine import error: {e}")
        return {
            "messages": [AIMessage(content="I'm having trouble accessing the art database. Let's continue our conversation and I'll try again later.")],
            "current_phase": "continuing"
        }
    
    # 3. Execute pipeline with progress updates
    try:
        start_time = time.time()
        
        # Step 5: Research brief (timeout: 30s)
        brief = await asyncio.wait_for(
            rag_session.generate_brief(curation_input['situation'], curation_input['emotions']),
            timeout=30.0
        )
        
        # Stage A: Candidate collection (timeout: 15s)
        candidates_result = await asyncio.wait_for(
            stage_a.collect_candidates(curation_input['situation'], curation_input['emotions']),
            timeout=15.0
        )
        
        # Extract candidate IDs and convert to format expected by Step 6
        candidate_ids = candidates_result.get("final_stageA_ids", [])
        candidates = [{"artwork_id": cid} for cid in candidate_ids]
        
        # Step 6: Reranking to top 8 for conversation (timeout: 45s)
        target_count = 8  # Fewer than 30 for conversational flow
        final_recs = await asyncio.wait_for(
            reranker.rerank_candidates(brief, candidates, target_count),
            timeout=45.0
        )
        
        processing_time = time.time() - start_time
        
        # 4. Format for conversation
        response_text = format_recommendations_for_chat(final_recs, curation_input, processing_time)
        
        # 5. Background memory update (non-blocking)
        # TODO: Implement update_art_preferences_after_curation in Step 2.3
        
        return {
            "messages": [AIMessage(content=response_text)],
            "last_curation_result": final_recs,
            "current_phase": "continuing",
            "curation_metadata": {
                "processing_time": processing_time,
                "confidence_level": curation_input["confidence_level"],
                "preferences_used": bool(curation_input["preferences"])
            }
        }
        
    except asyncio.TimeoutError:
        timeout_msg = ("I'm still working on finding the perfect artwork for you. "
                      "Let's continue our conversation while I process that in the background.")
        return {
            "messages": [AIMessage(content=timeout_msg)],
            "current_phase": "continuing"
        }
        
    except Exception as e:
        print(f"Art curation error: {e}")
        import traceback
        traceback.print_exc()
        error_msg = ("I encountered an issue with the art database. "
                    "Would you like to continue our conversation, and I can try the recommendations again later?")
        return {
            "messages": [AIMessage(content=error_msg)],
            "current_phase": "continuing"
        }
    
    finally:
        # Always restore original working directory
        os.chdir(original_cwd)
        print(f"DEBUG: Restored working directory to {original_cwd}")

async def art_curation_node(state: EmotionalArtState, config: RunnableConfig, store: BaseStore):
    """Art curation dedicated node - Art Curation Engine execution"""
    
    # Add store validation and fallback handling
    if store is None:
        print("WARNING: Store is None in art_curation_node, using InMemoryStore fallback")
        from langgraph.store.memory import InMemoryStore
        store = InMemoryStore()
    
    try:
        print(f"DEBUG: Starting art curation node")
        
        # 1. Load user memories
        user_id = config.get("configurable", {}).get("user_id", "default")
        namespace = ("memory", user_id)
        user_memories = await store.asearch(namespace)
        memories_dict = {item.key: item.value for item in user_memories}
        print(f"DEBUG: Loaded user memories: {list(memories_dict.keys())}")
        
        # 2. Prepare curation input
        curation_input = prepare_curation_input(memories_dict, state)
        print(f"DEBUG: Curation input prepared: {curation_input}")
        
        # 3. Set working directory to art_curation_engine
        original_cwd = await asyncio.to_thread(os.getcwd)
        # Use absolute path - ensure we get the project root directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # If we're already in art_curation_engine, go up one level
        if script_dir.endswith('art_curation_engine'):
            project_root = os.path.dirname(script_dir)
        else:
            project_root = script_dir
        art_engine_path = os.path.join(project_root, "art_curation_engine")
        
        if os.path.exists(art_engine_path):
            os.chdir(art_engine_path)
            print(f"DEBUG: Changed to art curation directory: {art_engine_path}")
        else:
            print(f"DEBUG: Art curation directory not found: {art_engine_path}")
            return {
                "messages": [AIMessage(content="I'm having trouble accessing the art database. Let's continue our conversation.")],
                "current_phase": "continuing"
            }
        
        # 4. Import and initialize (lazy loading for performance)
        try:
            # Add project root to Python path for imports
            import sys
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from art_curation_engine.core import RAGSessionBrief, StageACollector, Step6LLMReranker
            
            rag_session = RAGSessionBrief()
            stage_a = StageACollector()
            reranker = Step6LLMReranker(
            batch_size=2,  # Reduced batch size
            max_workers=1,  # Single worker to prevent concurrent calls
            api_delay_seconds=3.0,  # 3 second delay between API calls
            max_retries=3  # Retry failed calls up to 3 times
        )
            
        except ImportError as e:
            print(f"Art curation engine import error: {e}")
            return {
                "messages": [AIMessage(content="I'm having trouble accessing the art database. Let's continue our conversation and I'll try again later.")],
                "current_phase": "continuing"
            }
        
        # 5. Execute pipeline with progress updates
        try:
            import time
            start_time = time.time()
            
            print(f"DEBUG: Starting Step 5 - RAG brief generation")
            # Step 5: Research brief (run in thread pool)
            loop = asyncio.get_event_loop()
            brief = await loop.run_in_executor(
                None, 
                rag_session.generate_brief, 
                curation_input['situation'], 
                curation_input['emotions']
            )
            
            print(f"DEBUG: Starting Stage A - Candidate collection")
            # Stage A: Candidate collection (run in thread pool)
            candidates_result = await loop.run_in_executor(
                None,
                stage_a.collect_candidates,
                curation_input['situation'], 
                curation_input['emotions']
            )
            
            # Extract candidate IDs and convert to format expected by Step 6
            candidate_ids = candidates_result.get("final_stageA_ids", [])
            candidates = [{"artwork_id": cid} for cid in candidate_ids]
            print(f"DEBUG: Extracted {len(candidate_ids)} candidate IDs from Stage A")
            
            print(f"DEBUG: Starting Step 6 - LLM reranking")
            # Step 6: Reranking to top 8 for conversation (run in thread pool)
            target_count = 8  # Fewer than 30 for conversational flow
            final_recs = await loop.run_in_executor(
                None,
                reranker.rerank_candidates,
                brief, candidates, target_count
            )
            
            processing_time = time.time() - start_time
            print(f"DEBUG: Art curation completed in {processing_time:.2f}s")
            
            # 5. Format for conversation
            response_text = format_recommendations_for_chat(final_recs, curation_input, processing_time)
            
            # 6. Background memory update (non-blocking)
            # TODO: Implement update_art_preferences_after_curation in Step 2.3
            
            return {
                "messages": [AIMessage(content=response_text)],
                "last_curation_result": final_recs,
                "current_phase": "post_curation",
                "consent_for_reco": False,  # Reset consent to prevent infinite loop
                "curation_metadata": {
                    "processing_time": processing_time,
                    "confidence_level": curation_input["confidence_level"],
                    "preferences_used": bool(curation_input["preferences"])
                }
            }
            
        except asyncio.TimeoutError:
            print(f"DEBUG: Art curation timeout")
            timeout_msg = ("I'm still working on finding the perfect artwork for you. "
                          "Let's continue our conversation while I process that in the background.")
            return {
                "messages": [AIMessage(content=timeout_msg)],
                "current_phase": "continuing",
                "consent_for_reco": False  # Reset consent to prevent infinite loop
            }
            
        except Exception as e:
            print(f"DEBUG: Art curation error: {e}")
            import traceback
            traceback.print_exc()
            error_msg = ("I encountered an issue with the art database. "
                        "Would you like to continue our conversation, and I can try the recommendations again later?")
            return {
                "messages": [AIMessage(content=error_msg)],
                "current_phase": "continuing",
                "consent_for_reco": False  # Reset consent to prevent infinite loop
            }
        
        finally:
            # Always restore original working directory
            os.chdir(original_cwd)
            print(f"DEBUG: Restored working directory to {original_cwd}")
            
    except Exception as e:
        print(f"DEBUG: Error in art_curation_node: {e}")
        import traceback
        traceback.print_exc()
        return {
            "messages": [AIMessage(content="I'm having trouble with the art recommendations right now. Let's keep chatting!")],
            "current_phase": "continuing",
            "consent_for_reco": False  # Reset consent to prevent infinite loop
        }

# =============================================================================
# ROUTING
# =============================================================================

def should_curate_art(state: EmotionalArtState) -> Literal["art_curation_node", END]:
    """Conditional routing to determine if art curation is needed"""
    
    current_phase = state.get("current_phase", "")
    
    # Execute art curation only in ready_for_curation phase or when user consents
    if (current_phase == "ready_for_curation" or 
        state.get("consent_for_reco", False)):
        print(f"DEBUG: Routing to art curation - phase: {current_phase}, consent: {state.get('consent_for_reco', False)}")
        return "art_curation_node"
    else:
        print(f"DEBUG: Routing to END - phase: {current_phase}, consent: {state.get('consent_for_reco', False)}")
        return END


def route_message(state: EmotionalArtState, config: RunnableConfig, store: BaseStore) -> Literal[END, "update_user_profile", "update_art_preferences", "update_interaction_patterns"]:
    """Message routing - determine memory update node based on tool call"""
    
    message = state['messages'][-1]
    
    if not hasattr(message, 'tool_calls') or len(message.tool_calls) == 0:
        return END
    
    tool_call = message.tool_calls[0]
    update_type = tool_call['args']['update_type']
    
    if update_type == "profile":
        return "update_user_profile"
    elif update_type == "art_preferences":
        return "update_art_preferences"
    elif update_type == "interaction_patterns":
        return "update_interaction_patterns"
    else:
        print(f"Unknown update_type: {update_type}")
        return END

# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

# Create StateGraph
builder = StateGraph(EmotionalArtState)

# Add nodes
builder.add_node("conversation_node", conversation_node)
builder.add_node("art_curation_node", art_curation_node)

# Define edges
builder.add_edge(START, "conversation_node")
builder.add_conditional_edges(
    "conversation_node",
    should_curate_art,
    {
        "art_curation_node": "art_curation_node",
        END: END
    }
)
# After art curation, end the turn - user's next message will start a new turn
builder.add_edge("art_curation_node", END)

# Graph Compile - LangGraph API handles persistence automatically
graph = builder.compile()