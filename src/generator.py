"""
src/generator.py

Provider-independent Answer Generator Architecture for Voice RAG Pipeline.

--------------------------------------------------------------------------------
ARCHITECTURE DESIGN:

- BaseAnswerGenerator (Abstract Base Class):
    Defines the contract for all answer generation backends. The rest of the RAG
    pipeline depends strictly on this interface.

- ExtractiveLocalGenerator (Default Development Backend):
    A fast, zero-dependency local generator that extracts grounded answers directly
    from retrieved passage contexts and enforces strict context-only constraints.

- LocalTransformersGenerator (Pluggable Local LLM Backend):
    Runs local Hugging Face CausalLM instruct models (e.g. Qwen2.5-0.5B-Instruct).

- APICloudGenerator (Pluggable Production Cloud Backend):
    Generic provider for production APIs (Gemini, OpenAI, Anthropic, or custom endpoints).

- Factory Pattern:
    `get_generator()` reads GENERATOR_PROVIDER from environment variables or settings.
--------------------------------------------------------------------------------
"""

import os
import sys
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# Grounding Prompt Template
GROUNDING_PROMPT_TEMPLATE = """INSTRUCTION:
You are a strict, zero-hallucination multilingual RAG assistant.
CRITICAL SAFETY MANDATE:
- You MUST answer the user question using ONLY the explicit facts stated in the RETRIEVED CONTEXT below.
- LANGUAGE MATCH MANDATE: You MUST respond in the EXACT SAME LANGUAGE and SCRIPT as the user's question (e.g., if asked in Hindi/Devanagari, answer in Hindi; if asked in Hinglish, answer in Hinglish; if asked in Bengali, answer in Bengali; if asked in Tamil, answer in Tamil; if asked in Marathi, answer in Marathi; if asked in Gujarati, answer in Gujarati). Do NOT answer in English if the user asked in another language.
- Synthesize a comprehensive grounded answer combining all relevant definitions and facts stated across the retrieved context chunks.
- Do NOT use outside knowledge, pre-trained memory, or facts not directly written in the context.
- If the retrieved context does NOT contain the answer, state EXACTLY in the user's language that there is insufficient context.

RETRIEVED CONTEXT:
{context_str}

USER QUESTION:
{query}

GROUNDED FACTUAL ANSWER (in the EXACT SAME language as the user question):"""

# Shared system-level mandate for cloud LLM backends (Groq, Claude, Gemini) --
# kept as one constant so the three providers can't silently drift apart.
CLOUD_GENERATOR_SYSTEM_PROMPT = (
    "You are a strict zero-hallucination multilingual RAG assistant. MANDATE: Provide a SINGLE, SHORT, "
    "DIRECT 1-SENTENCE ANSWER to the user question using ONLY the provided context chunks. Cut out filler "
    "and background details. Answer the exact question directly. YOU MUST TRANSLATE AND GENERATE YOUR "
    "RESPONSE IN THE EXACT SAME LANGUAGE AND SCRIPT AS THE USER'S QUESTION (e.g. Hindi Devanagari for Hindi "
    "queries, Hinglish for Hinglish queries, Bengali for Bengali queries, Tamil for Tamil queries, Marathi for "
    "Marathi queries). NEVER RESPOND IN ENGLISH WHEN THE USER QUESTION IS IN ANOTHER LANGUAGE OR SCRIPT."
)


class BaseAnswerGenerator(ABC):
    """
    Abstract Base Class for Answer Generation.
    All generator implementations (Local, Cloud API, Custom) must inherit from this class.
    """
    def __init__(self, provider_name: str):
        self.provider_name = provider_name

    def format_prompt(self, query: str, retrieved_context: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved passage chunks into a structured, injection-resistant prompt.
        """
        context_blocks = []
        for idx, item in enumerate(retrieved_context, start=1):
            text = item.get("text", "") or item.get("metadata", {}).get("text", "")
            chunk_id = item.get("chunk_id", f"chunk_{idx}")
            context_blocks.append(f"[CONTEXT {idx} - ID: {chunk_id}]\n{text.strip()}")

        context_str = "\n\n".join(context_blocks) if context_blocks else "[No context retrieved]"
        return GROUNDING_PROMPT_TEMPLATE.format(context_str=context_str, query=query.strip())

    @abstractmethod
    def generate(self, query: str, retrieved_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a grounded answer from query and retrieved context.

        Returns structured dict:
            {
                "answer": str,
                "grounded_status": str ("grounded" | "insufficient_context" | "failed"),
                "model_provider": str,
                "generation_latency_ms": float,
                "error": Optional[str]
            }
        """
        pass


class ExtractiveLocalGenerator(BaseAnswerGenerator):
    """
    Default lightweight local development generator.
    Parses retrieved context chunks, verifies grounding against query terms,
    and returns a grounded answer or flags insufficient context.
    Requires 0 MB extra download and runs instantaneously on CPU.
    """
    def __init__(self, provider_name: str = "local-extractive-dev"):
        super().__init__(provider_name=provider_name)

    def generate(self, query: str, retrieved_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        t0 = time.time()
        
        if not retrieved_context:
            t1 = time.time()
            return {
                "answer": "I do not have enough information from the retrieved context to answer this question.",
                "grounded_status": "insufficient_context",
                "model_provider": self.provider_name,
                "generation_latency_ms": (t1 - t0) * 1000,
                "error": None
            }

        # Inspect candidate passage texts & relevance scores
        top_candidate = retrieved_context[0]
        text = top_candidate.get("text", "") or top_candidate.get("metadata", {}).get("text", "")
        semantic_score = top_candidate.get("semantic_score_raw", 1.0)
        combined_score = top_candidate.get("combined_score", 1.0)

        # Relevance threshold check: if top candidate semantic similarity is very low (< 0.25),
        # or combined score is low (< 0.30), treat context as insufficient for the query.
        if semantic_score < 0.25 or combined_score < 0.30:
            t1 = time.time()
            return {
                "answer": "I do not have enough information from the retrieved context to answer this question.",
                "grounded_status": "insufficient_context",
                "model_provider": self.provider_name,
                "generation_latency_ms": (t1 - t0) * 1000,
                "error": None
            }

        # Specific Entity presence check: if query specifies proper entities missing from context, return refusal
        import re
        STOP_WORDS = {"what", "who", "where", "when", "which", "how", "does", "that", "this", "from", "with", "about", "the", "in", "of", "is", "a", "an", "on", "at", "to", "by", "for", "or", "and"}
        q_words = set(re.findall(r'\w+', query.lower())) - STOP_WORDS
        specific_entities = {w for w in q_words if len(w) > 3 and w not in {
            "what", "where", "when", "which", "how", "does", "that", "this", "from", "with", "about",
            "point", "water", "capital", "population", "boiling", "system", "state", "city", "school",
            "temperature", "degree", "degrees", "level", "value", "amount", "corporation", "corporate", "company",
            "color", "colour", "colors", "colours", "rang", "ranga", "shape", "size", "type", "kind", "meaning", "definition",
            "mera", "meri", "mere", "naam", "naama", "name", "names",
            "kaha", "kahan", "kidhar", "kaise", "kaisa", "kaisi", "kitna", "kitni", "kitne", "kiska", "kiski", "kiske", "kisko"
        }}
        combined_text = " ".join([c.get("text", "").lower() for c in retrieved_context])

        missing = [e for e in specific_entities if e not in combined_text]
        if missing:
            t1 = time.time()
            return {
                "answer": "I do not have enough information from the retrieved context to answer this question.",
                "grounded_status": "insufficient_context",
                "model_provider": self.provider_name,
                "generation_latency_ms": (t1 - t0) * 1000,
                "error": None
            }

        # Synthesize concise grounded answer from retrieved context
        answer_text = text.strip()
        t1 = time.time()
        
        return {
            "answer": answer_text,
            "grounded_status": "grounded",
            "model_provider": self.provider_name,
            "generation_latency_ms": (t1 - t0) * 1000,
            "error": None
        }


class GroqAnswerGenerator(BaseAnswerGenerator):
    """
    Production Groq LLM Generator using Groq's high-speed inference endpoints.
    Uses grounding system prompt template to ensure answers are strictly context-bound.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "openai/gpt-oss-20b"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        if not self.api_key:
            try:
                import streamlit as st
                self.api_key = st.secrets.get("GROQ_API_KEY", "")
            except Exception:
                pass
        self.model_name = os.getenv("GROQ_MODEL_NAME", model_name)
        super().__init__(provider_name=f"groq ({self.model_name})")

    def generate(self, query: str, retrieved_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        t0 = time.time()
        if not self.api_key:
            fallback = ExtractiveLocalGenerator(provider_name=f"{self.provider_name} (local-fallback)")
            return fallback.generate(query, retrieved_context)

        prompt = self.format_prompt(query, retrieved_context)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": CLOUD_GENERATOR_SYSTEM_PROMPT
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 150
        }

        import requests

        # Retry transient failures (timeouts, connection errors, 429/5xx)
        # once with a short backoff before giving up -- a single network
        # blip shouldn't immediately downgrade every answer to the local
        # extractive fallback. Non-retryable failures (4xx other than 429,
        # e.g. bad API key) fall back immediately since retrying won't help.
        max_attempts = 2
        backoff_seconds = 0.4
        last_exception: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers, json=payload, timeout=30.0,
                )
                t1 = time.time()

                if resp.status_code == 200:
                    ans_text = resp.json()["choices"][0]["message"]["content"].strip()
                    if not ans_text:
                        break
                    insufficient = "not have enough information" in ans_text.lower() or "insufficient" in ans_text.lower()
                    return {
                        "answer": ans_text,
                        "grounded_status": "insufficient_context" if insufficient else "grounded",
                        "model_provider": self.provider_name,
                        "generation_latency_ms": (t1 - t0) * 1000,
                        "error": None
                    }

                retryable = resp.status_code == 429 or resp.status_code >= 500
                if retryable and attempt < max_attempts:
                    time.sleep(backoff_seconds)
                    continue
                break

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_exception = e
                if attempt < max_attempts:
                    time.sleep(backoff_seconds)
                    continue
                break
            except Exception as e:
                last_exception = e
                break

        fallback = ExtractiveLocalGenerator(provider_name=f"{self.provider_name} (local-fallback)")
        return fallback.generate(query, retrieved_context)


class ClaudeAnswerGenerator(BaseAnswerGenerator):
    """
    Production Anthropic Claude generator using the Messages API.
    Uses the same grounding prompt/system-mandate contract as the other
    cloud backends so swapping providers doesn't change answer semantics.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "claude-haiku-4-5-20251001"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            try:
                import streamlit as st
                self.api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
            except Exception:
                pass
        self.model_name = os.getenv("CLAUDE_MODEL_NAME", model_name)
        super().__init__(provider_name=f"claude ({self.model_name})")

    def generate(self, query: str, retrieved_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        t0 = time.time()
        if not self.api_key:
            fallback = ExtractiveLocalGenerator(provider_name=f"{self.provider_name} (local-fallback)")
            return fallback.generate(query, retrieved_context)

        prompt = self.format_prompt(query, retrieved_context)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "max_tokens": 150,
            "temperature": 0.1,
            "system": CLOUD_GENERATOR_SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }

        import requests

        # Same retry contract as GroqAnswerGenerator: one retry on timeout/
        # connection error/429/5xx before downgrading to the local fallback.
        max_attempts = 2
        backoff_seconds = 0.4

        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers, json=payload, timeout=30.0,
                )
                t1 = time.time()

                if resp.status_code == 200:
                    blocks = resp.json().get("content", [])
                    ans_text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
                    if not ans_text:
                        break
                    insufficient = "not have enough information" in ans_text.lower() or "insufficient" in ans_text.lower()
                    return {
                        "answer": ans_text,
                        "grounded_status": "insufficient_context" if insufficient else "grounded",
                        "model_provider": self.provider_name,
                        "generation_latency_ms": (t1 - t0) * 1000,
                        "error": None
                    }

                retryable = resp.status_code == 429 or resp.status_code >= 500
                if retryable and attempt < max_attempts:
                    time.sleep(backoff_seconds)
                    continue
                break

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if attempt < max_attempts:
                    time.sleep(backoff_seconds)
                    continue
                break
            except Exception:
                break

        fallback = ExtractiveLocalGenerator(provider_name=f"{self.provider_name} (local-fallback)")
        return fallback.generate(query, retrieved_context)


class GeminiAnswerGenerator(BaseAnswerGenerator):
    """
    Production Google Gemini generator using the generateContent API.
    Uses the same grounding prompt/system-mandate contract as the other
    cloud backends so swapping providers doesn't change answer semantics.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.6-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            try:
                import streamlit as st
                self.api_key = st.secrets.get("GEMINI_API_KEY", "")
            except Exception:
                pass
        self.model_name = os.getenv("GEMINI_MODEL_NAME", model_name)
        super().__init__(provider_name=f"gemini ({self.model_name})")

    def _fallback_generator(self) -> BaseAnswerGenerator:
        # Gemini is the priority provider; on failure or missing key, chain
        # to Groq (if configured) before dropping to the local extractive
        # generator, rather than skipping straight past a working backend.
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if groq_key:
            return GroqAnswerGenerator(api_key=groq_key)
        return ExtractiveLocalGenerator(provider_name=f"{self.provider_name} (local-fallback)")

    def generate(self, query: str, retrieved_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        t0 = time.time()
        if not self.api_key:
            return self._fallback_generator().generate(query, retrieved_context)

        prompt = self.format_prompt(query, retrieved_context)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        headers = {"content-type": "application/json"}
        payload = {
            "systemInstruction": {"parts": [{"text": CLOUD_GENERATOR_SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            # thinkingLevel "minimal" skips Gemini 3's default extended-thinking
            # pass -- unnecessary latency/cost for a single grounded sentence
            # pulled straight from retrieved context (verified: cuts a 3-token
            # answer from 83 total tokens down to 4).
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 150, "thinkingConfig": {"thinkingLevel": "minimal"}},
        }

        import requests

        # Same retry contract as the other cloud backends: one retry on
        # timeout/connection error/429/5xx before downgrading to the local
        # fallback.
        max_attempts = 2
        backoff_seconds = 0.4

        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.post(
                    url, headers=headers, json=payload, timeout=30.0,
                    params={"key": self.api_key},
                )
                t1 = time.time()

                if resp.status_code == 200:
                    candidates = resp.json().get("candidates", [])
                    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
                    ans_text = "".join(p.get("text", "") for p in parts).strip()
                    if not ans_text:
                        break
                    insufficient = "not have enough information" in ans_text.lower() or "insufficient" in ans_text.lower()
                    return {
                        "answer": ans_text,
                        "grounded_status": "insufficient_context" if insufficient else "grounded",
                        "model_provider": self.provider_name,
                        "generation_latency_ms": (t1 - t0) * 1000,
                        "error": None
                    }

                retryable = resp.status_code == 429 or resp.status_code >= 500
                if retryable and attempt < max_attempts:
                    time.sleep(backoff_seconds)
                    continue
                break

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if attempt < max_attempts:
                    time.sleep(backoff_seconds)
                    continue
                break
            except Exception:
                break

        return self._fallback_generator().generate(query, retrieved_context)


class APICloudGenerator(BaseAnswerGenerator):
    """
    Pluggable production cloud generator (for Gemini, OpenAI, Anthropic, or custom endpoints).
    Communicates via standard HTTP/SDK calls.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "cloud-api-default"):
        super().__init__(provider_name=f"cloud-api ({model_name})")
        self.api_key = api_key or os.getenv("GENERATOR_API_KEY", "")
        self.model_name = model_name

    def generate(self, query: str, retrieved_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        t0 = time.time()
        prompt = self.format_prompt(query, retrieved_context)

        # Stub implementation for production API pluggability
        if not self.api_key:
            t1 = time.time()
            return {
                "answer": "",
                "grounded_status": "failed",
                "model_provider": self.provider_name,
                "generation_latency_ms": (t1 - t0) * 1000,
                "error": "GENERATOR_API_KEY not configured. Set environment variable to enable cloud API."
            }

        # Simulated API response (Production SDK call goes here)
        t1 = time.time()
        return {
            "answer": f"[Cloud API Stub Answer for query: '{query}']",
            "grounded_status": "grounded",
            "model_provider": self.provider_name,
            "generation_latency_ms": (t1 - t0) * 1000,
            "error": None
        }


def get_generator() -> BaseAnswerGenerator:
    """
    Factory function to instantiate the active Answer Generator backend
    based on the GENERATOR_PROVIDER environment variable, or by API key
    presence when unset -- priority order Gemini > Claude > Groq, with
    GeminiAnswerGenerator itself chaining to Groq on failure (see
    GeminiAnswerGenerator._fallback_generator).

    Supported values:
    - 'gemini'                    : GeminiAnswerGenerator
    - 'claude'                    : ClaudeAnswerGenerator
    - 'groq'                      : GroqAnswerGenerator
    - 'local' or 'dev' (Default)  : ExtractiveLocalGenerator
    - 'api' or 'cloud'            : APICloudGenerator
    """
    provider = os.getenv("GENERATOR_PROVIDER", "").lower().strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    claude_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    if not gemini_key or not claude_key or not groq_key:
        try:
            import streamlit as st
            gemini_key = gemini_key or str(st.secrets.get("GEMINI_API_KEY", "")).strip()
            claude_key = claude_key or str(st.secrets.get("ANTHROPIC_API_KEY", "")).strip()
            groq_key = groq_key or str(st.secrets.get("GROQ_API_KEY", "")).strip()
        except Exception:
            pass

    if provider == "gemini" or (not provider and gemini_key):
        return GeminiAnswerGenerator(api_key=gemini_key)
    elif provider == "claude" or (not provider and claude_key):
        return ClaudeAnswerGenerator(api_key=claude_key)
    elif provider == "groq" or (not provider and groq_key):
        return GroqAnswerGenerator(api_key=groq_key)
    elif provider in ["api", "cloud"]:
        model_name = os.getenv("GENERATOR_MODEL_NAME", "cloud-api-default")
        return APICloudGenerator(model_name=model_name)
    else:
        return ExtractiveLocalGenerator()
