"""
src/guardrails.py

Comprehensive Guardrail Layer for Voice RAG Pipeline.

--------------------------------------------------------------------------------
ARCHITECTURE:

User Query
    ↓
InputGuardrail (Empty / Off-topic / Safety checks)
    ↓ (If allowed)
Hybrid Retrieval (FAISS + BM25)
    ↓
ContextValidator (Context quality & threshold validation)
    ↓ (If sufficient)
Answer Generator
    ↓
GroundingGuardrail (Hallucination & context adherence validation)
    ↓
Final Response
--------------------------------------------------------------------------------
"""

import re
import sys
import time
from typing import List, Dict, Any, Optional

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# --- Configurable Off-Topic & Injection Patterns ---
# Covers direct override phrasing ("ignore/disregard/forget ... instructions")
# as well as the "forget X, do Y instead" redirect pattern, which is a
# distinct bypass: it doesn't ask to reveal or override rules explicitly, it
# just tries to substitute a new task for the original one.
# Whole-query chit-chat/greeting patterns (anchored, not substring search).
# A bare "Hello, how are you?" retrieves *something* from a 16k-chunk corpus
# by cosine similarity no matter how irrelevant -- text like a stray "Hello,
# internet!" chunk can score high purely on lexical overlap with "hello"
# -- and without this check it sails past ContextValidator's relevance
# threshold into a real (slow, ~1-4s) LLM call that either echoes that junk
# back as a nonsense "answered" response or eventually self-refuses anyway.
# Catching it here rejects it in <1ms instead, before retrieval ever runs.
CHITCHAT_PATTERNS = [
    r"^(hi|hello|hey|hiya|yo)( there)?[\s!.,]*$",
    r"^(hi|hello|hey|hiya|yo)[\s,!.]*how\s+are\s+you( doing)?[\s?!.]*$",
    r"^how\s+are\s+you( doing)?[\s?!.]*$",
    r"^good\s+(morning|afternoon|evening|night)[\s!.,]*$",
    r"^(what'?s\s+up|sup)[\s?!.]*$",
    r"^(thanks?|thank\s+you|thx)(\s+(so\s+much|a\s+lot|very\s+much))?[\s!.,]*$",
    r"^(bye|goodbye|see\s+you|see\s+ya)[\s!.,]*$",
    r"^who\s+are\s+you[\s?!.]*$",
]

OFF_TOPIC_PATTERNS = [
    r"\b(write|compose|create|generate)\s+.*(poem|song|story|essay|joke|game|script|code|program|function)\b",
    r"\b(tell|give)\s+me\s+.*(joke|riddle|story|poem)\b",
    r"\b(ignore|disregard|bypass|forget)\s+.*(previous|prior|above|system|prompt|instruction)",
    r"\b(forget|ignore)\s+.*(instead|and\s+(instead\s+)?(just\s+)?(tell|give|do|say))\b",
    r"\b(reveal|show|print|display|output|repeat)\s+.*(system\s+prompt|your\s+instructions|the\s+instructions)",
    r"\byou\s+are\s+now\s+(a|an|the)\b",
    r"\bact\s+as\s+(if\s+you|a\s+jailbroken|an?\s+unrestricted)\b",
    r"\b(play|start)\s+(a\s+)?game\b",
    r"\b(python|javascript|c\+\+|java)\s+(code|function|script|program)\b"
]

# --- Configurable Basic Safety Patterns ---
# Includes the direct request form as well as a fictional/roleplay-framing
# form ("for a story/novel/movie, describe/explain how to <harmful act>"),
# since wrapping a harmful request in a fictional pretext is a common and
# simple bypass of a purely literal "how to X" pattern.
UNSAFE_PATTERNS = [
    r"\b(make|build|create|manufacture|construct\w*)\s+.*(bomb|weapon|explosive|poison|harmful)\b",
    r"\b(hack|crack|breach|exploit)\s+.*(system|password|account)\b",
    r"\b(how\s+to|best\s+way\s+to|ways?\s+to)\s+(commit\s+)?suicide\b",
    r"\bself[\s-]harm\b",
    r"\bkill\s+(myself|yourself|himself|herself|themselves)\b",
    r"\b(illegal|stolen|dangerous\s+weapon)\b",
    r"\b(for\s+(a|my)\s+(story|novel|movie|script|game|thriller|character)).{0,60}"
    r"\b(describe|explain|detail|show|write)\b.{0,40}"
    r"\b(bomb|explosive|weapon|poison|kill|murder|hack)\w*\b",
    r"\b(step[\s-]by[\s-]step|detailed)\s+.*(bomb|explosive|weapon|poison)\b",
]


class InputGuardrail:
    """
    Validates user queries before embedding and retrieval.
    Rejects empty inputs, off-topic requests, and unsafe content.
    Multilingual-safe: preserves non-English (Indic) queries.
    """
    def __init__(
        self,
        off_topic_patterns: Optional[List[str]] = None,
        unsafe_patterns: Optional[List[str]] = None,
        chitchat_patterns: Optional[List[str]] = None
    ):
        self.off_topic_patterns = [re.compile(p, re.IGNORECASE) for p in (off_topic_patterns or OFF_TOPIC_PATTERNS)]
        self.unsafe_patterns = [re.compile(p, re.IGNORECASE) for p in (unsafe_patterns or UNSAFE_PATTERNS)]
        self.chitchat_patterns = [re.compile(p, re.IGNORECASE) for p in (chitchat_patterns or CHITCHAT_PATTERNS)]

    def validate(self, query: str) -> Dict[str, Any]:
        if not query or not query.strip():
            return {
                "allowed": False,
                "reason": "Query is empty or contains only whitespace.",
                "category": "invalid"
            }

        q_clean = query.strip()

        # A query made only of punctuation/symbols ("?", "...") has no
        # actual question in it -- letting it through wastes a retrieval
        # call on nothing and can coincidentally "answer" from whatever
        # scores highest by default.
        if not re.search(r"\w", q_clean, flags=re.UNICODE):
            return {
                "allowed": False,
                "reason": "Query contains no actual words to search for.",
                "category": "invalid"
            }

        # Check chit-chat/greetings -- these can retrieve *something* from
        # the corpus by incidental lexical overlap and would otherwise slip
        # past the relevance threshold into a real, slow LLM call. Checked
        # before the (substring-search) off-topic patterns since this is a
        # whole-query match, cheaper and more precise for this case.
        for pattern in self.chitchat_patterns:
            if pattern.match(q_clean):
                return {
                    "allowed": False,
                    "reason": "Query is a greeting/chit-chat, not a question this RAG system can answer from its corpus.",
                    "category": "off_topic"
                }

        # Check safety violations
        for pattern in self.unsafe_patterns:
            if pattern.search(q_clean):
                return {
                    "allowed": False,
                    "reason": "Query contains unsafe or restricted content.",
                    "category": "unsafe"
                }

        # Check off-topic non-RAG requests
        for pattern in self.off_topic_patterns:
            if pattern.search(q_clean):
                return {
                    "allowed": False,
                    "reason": "Query is off-topic (creative writing, coding, or non-RAG request).",
                    "category": "off_topic"
                }

        return {
            "allowed": True,
            "reason": "Query passed input validation.",
            "category": "allowed"
        }


class ContextValidator:
    """
    Evaluates retrieved passage quality and score thresholds before calling the answer generator.
    """
    def __init__(self, min_combined_score: float = 0.25, min_semantic_score: float = 0.20):
        self.min_combined_score = min_combined_score
        self.min_semantic_score = min_semantic_score

    def validate_context(self, retrieved_candidates: List[Dict[str, Any]], query: str = "") -> Dict[str, Any]:
        if not retrieved_candidates:
            return {
                "sufficient": False,
                "reason": "No context passages retrieved.",
                "status": "insufficient_context"
            }

        top_candidate = retrieved_candidates[0]
        comb_score = top_candidate.get("combined_score", top_candidate.get("score", 0.0))
        sem_score = top_candidate.get("semantic_score_raw", top_candidate.get("score", 0.0))

        if comb_score < self.min_combined_score:
            return {
                "sufficient": False,
                "reason": f"Top candidate combined score ({comb_score:.3f}) below relevance threshold ({self.min_combined_score}).",
                "status": "insufficient_context"
            }
        if sem_score < self.min_semantic_score:
            return {
                "sufficient": False,
                "reason": f"Top candidate semantic score ({sem_score:.3f}) below relevance threshold ({self.min_semantic_score}).",
                "status": "insufficient_context"
            }

        # Subject Entity & Keyword Presence Validation
        if query and query.strip():
            # Stop words including English & Indic transliterated connectors
            STOP_WORDS = {
                "what", "who", "where", "when", "which", "how", "does", "that", "this", "from", "with",
                "about", "the", "in", "of", "is", "a", "an", "on", "at", "to", "by", "for", "or", "and",
                "ka", "ki", "ke", "ko", "me", "par", "se", "h", "hai", "hain", "kitna", "kitne", "kitni", "hota", "hoti", "hote", "kya"
            }
            q_words = set(re.findall(r'\w+', query.lower())) - STOP_WORDS
            
            # Combine text across top candidates
            combined_candidate_text = " ".join([c.get("text", "").lower() for c in retrieved_candidates])

            # ASCII entity presence can only ever match when the retrieved
            # passages are themselves in a Latin script. For a cross-lingual
            # match (e.g. an English query answered from an Indic-script
            # translated passage), no ASCII substring can appear in the
            # passage text by definition, so this check would reject every
            # correct cross-lingual answer. Skip it when the candidate text
            # is predominantly non-Latin; the semantic + BM25 hybrid score
            # already validated relevance above.
            letters = [c for c in combined_candidate_text if c.isalpha()]
            ascii_letter_ratio = (sum(1 for c in letters if ord(c) < 128) / len(letters)) if letters else 1.0
            if ascii_letter_ratio < 0.3:
                return {
                    "sufficient": True,
                    "reason": "Context passes relevance quality checks (entity check skipped: non-Latin script passage).",
                    "status": "sufficient"
                }

            # Identify genuine named entities (proper nouns, years) to check for
            # presence, rather than every uncommon-looking word. A blocklist of
            # excluded common words is fragile -- it under-blocks (any common verb
            # not yet added, e.g. "wrote", gets treated as a required entity and
            # wrongly rejects a well-retrieved, correct answer) and still misses
            # words nobody thought to add. Instead: a word only counts as a
            # "specific entity" if it is capitalized in the *original* (non-
            # lowercased) query and isn't the first word of the sentence (which
            # is capitalized regardless of what it is), or if it's a standalone
            # 4-digit year/number. Both are strong, low-false-positive signals
            # that the word names a particular person/place/thing the answer
            # must actually be about.
            original_words = re.findall(r"\w+", query)
            capitalized_entities = {
                w.lower() for i, w in enumerate(original_words)
                if i > 0 and len(w) > 3 and w[0].isupper() and w.lower() not in STOP_WORDS
            }
            year_entities = {w for w in q_words if re.fullmatch(r"(19|20)\d{2}", w)}
            specific_entities = (capitalized_entities | year_entities) & set(
                w for w in q_words if all(ord(c) < 128 for c in w)
            )

            # If the query names specific proper entities, verify they're present in context
            missing_entities = [e for e in specific_entities if e not in combined_candidate_text]
            if missing_entities:
                return {
                    "sufficient": False,
                    "reason": f"Retrieved context is missing key query entities {missing_entities}.",
                    "status": "insufficient_context"
                }

        return {
            "sufficient": True,
            "reason": "Context passes relevance quality checks.",
            "status": "sufficient"
        }


# Function words carry no evidence about whether an answer's *content* came
# from the retrieved passages -- counting them inflates overlap since almost
# any answer and almost any passage of reasonable length share several of
# these regardless of topic. Excluding them is what makes the overlap ratio
# below mean something.
_GROUNDING_STOP_WORDS = {
    "what", "who", "where", "when", "which", "how", "does", "that", "this", "from", "with",
    "about", "the", "in", "of", "is", "are", "was", "were", "a", "an", "on", "at", "to", "by",
    "for", "or", "and", "it", "its", "as", "be", "been", "has", "have", "had", "not", "no",
    "ka", "ki", "ke", "ko", "me", "par", "se", "h", "hai", "hain", "hota", "hoti", "hote",
    "kya", "ek", "mein", "hi", "bhi", "aur", "yeh", "woh",
}


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
            if w not in _GROUNDING_STOP_WORDS and len(w) > 1}


class GroundingGuardrail:
    """
    Lightweight, deterministic output validator.
    Ensures generated answers are grounded in the retrieved passage text.
    """
    def __init__(self, min_overlap_ratio: float = 0.20):
        self.min_overlap_ratio = min_overlap_ratio

    def validate_grounding(self, answer: str, retrieved_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not answer or not answer.strip():
            return {
                "grounded": False,
                "confidence": 0.0,
                "reason": "Generated answer is empty.",
                "status": "ungrounded"
            }

        # Standard fallback string is always grounded
        insufficient_msg = "I do not have enough information from the retrieved context to answer this question."
        if insufficient_msg in answer:
            return {
                "grounded": True,
                "confidence": 1.0,
                "reason": "Correct insufficient context fallback response.",
                "status": "grounded"
            }

        # Combine text from retrieved context
        context_text = " ".join([
            c.get("text", "") or c.get("metadata", {}).get("text", "")
            for c in retrieved_context
        ]).lower()

        # Content-word overlap: what share of the answer's substantive
        # (non-stopword) words actually appear in the retrieved passages.
        # This is the real evidence check -- it replaces the previous
        # "any single word matched" and "any multilingual answer passes"
        # fallbacks, both of which made the gate trivially satisfiable
        # (almost every answer shares at least one common word with almost
        # any passage, and a fabricated multilingual answer was never
        # checked against evidence at all).
        answer_content_words = _content_words(answer)
        context_content_words = _content_words(context_text)

        if not answer_content_words:
            return {"grounded": True, "confidence": 1.0, "reason": "Short non-word answer.", "status": "grounded"}

        matched_words = answer_content_words & context_content_words
        overlap_ratio = len(matched_words) / len(answer_content_words)

        # Extract digits/numbers from answer
        answer_digits = set(re.findall(r'\d+', answer))
        context_digits = set(re.findall(r'\d+', context_text))
        digit_match = bool(answer_digits and answer_digits.issubset(context_digits))

        if overlap_ratio >= self.min_overlap_ratio or digit_match:
            return {
                "grounded": True,
                "confidence": round(max(overlap_ratio, 0.6 if digit_match else overlap_ratio), 3),
                "reason": f"Answer is grounded with {overlap_ratio*100:.1f}% content-word overlap.",
                "status": "grounded"
            }

        return {
            "grounded": False,
            "confidence": round(overlap_ratio, 3),
            "reason": f"Answer content-word overlap ({overlap_ratio*100:.1f}%) is below minimum grounding threshold ({self.min_overlap_ratio*100:.0f}%).",
            "status": "ungrounded"
        }


class GuardedRAGPipeline:
    """
    Orchestrates the full end-to-end RAG pipeline with Input, Context, and Output Guardrails.
    """
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
        self.input_guardrail = InputGuardrail()
        self.context_validator = ContextValidator(min_combined_score=0.30)
        self.grounding_guardrail = GroundingGuardrail(min_overlap_ratio=0.20)

    def process(self, query: str) -> Dict[str, Any]:
        t_start = time.time()

        # 1. Input Guardrail
        t0 = time.time()
        input_eval = self.input_guardrail.validate(query)
        t1 = time.time()
        input_ms = (t1 - t0) * 1000

        if not input_eval["allowed"]:
            t_end = time.time()
            cat = input_eval["category"]
            msg = "I cannot process this request because it is off-topic or inappropriate for this RAG system." \
                if cat in ["off_topic", "unsafe"] else "Please provide a valid question."
            
            return {
                "query": query,
                "status": "rejected",
                "answer": msg,
                "grounded": False,
                "guardrail_reason": input_eval["reason"],
                "retrieved_chunks": [],
                "latency": {
                    "input_guardrail_ms": round(input_ms, 2),
                    "retrieval_ms": 0.0,
                    "generation_ms": 0.0,
                    "grounding_ms": 0.0,
                    "total_ms": round((t_end - t_start) * 1000, 2)
                }
            }

        # 2. Hybrid Retrieval
        t2 = time.time()
        retrieval_data = self.retriever.retrieve(query, top_k=3)
        t3 = time.time()
        retrieval_ms = (t3 - t2) * 1000
        candidates = retrieval_data.get("results", [])

        # 3. Context Quality Validation
        context_eval = self.context_validator.validate_context(candidates)
        if not context_eval["sufficient"]:
            t_end = time.time()
            return {
                "query": query,
                "status": "insufficient_context",
                "answer": "I do not have enough information from the retrieved context to answer this question.",
                "grounded": True,
                "guardrail_reason": context_eval["reason"],
                "retrieved_chunks": candidates,
                "latency": {
                    "input_guardrail_ms": round(input_ms, 2),
                    "retrieval_ms": round(retrieval_ms, 2),
                    "generation_ms": 0.0,
                    "grounding_ms": 0.0,
                    "total_ms": round((t_end - t_start) * 1000, 2)
                }
            }

        # 4. Answer Generation
        t4 = time.time()
        gen_output = self.generator.generate(query, candidates)
        t5 = time.time()
        generation_ms = (t5 - t4) * 1000
        raw_answer = gen_output.get("answer", "")

        # 5. Output / Grounding Guardrail Validation
        t6 = time.time()
        ground_eval = self.grounding_guardrail.validate_grounding(raw_answer, candidates)
        t7 = time.time()
        grounding_ms = (t7 - t6) * 1000
        t_end = time.time()

        final_answer = raw_answer if ground_eval["grounded"] else \
            "I do not have enough information from the retrieved context to answer this question."

        return {
            "query": query,
            "status": "answered" if ground_eval["grounded"] else "insufficient_context",
            "answer": final_answer,
            "grounded": ground_eval["grounded"],
            "guardrail_reason": ground_eval["reason"],
            "retrieved_chunks": candidates,
            "latency": {
                "input_guardrail_ms": round(input_ms, 2),
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "grounding_ms": round(grounding_ms, 2),
                "total_ms": round((t_end - t_start) * 1000, 2)
            }
        }
