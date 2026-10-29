"""Answer generation with the retrieved passages injected as numbered context.

The response is constrained to a JSON schema (structured outputs) instead of free text,
because every field is something the evaluation measures:

* `answerable` -- the model's own abstention decision,
* `citations`  -- which passages the answer claims to rest on,
* `quotes`     -- verbatim supporting sentences, so that groundedness can be checked by
                  string matching against the cited passage instead of by another model.

A model that paraphrases inside `quotes` fails that check. This is intended: a quote
that cannot be found in the passage is exactly the kind of evidence an auditor would
reject.
"""

import json
from dataclasses import asdict, dataclass, field

from regrag import config
from regrag.retrieve import Hit

SYSTEM_PROMPT = """\
You answer questions about German banking supervision requirements. Your only source is \
the numbered passages in the user message, taken from BaFin circulars.

Rules:
1. Use only what the passages say. Do not add facts from outside the passages, even if \
you are confident they are correct; an answer that is right for the wrong reason is \
still a failure here.
2. If the passages do not contain the answer, set "answerable" to false and say in one \
or two sentences what is missing. Do not guess. If they answer only part of the \
question, answer that part and state what is not covered.
3. List in "citations" the numbers of every passage your answer relies on.
4. For each cited passage give at least one supporting quote in "quotes": copied \
character for character from that passage, one or two sentences, no ellipses, no \
paraphrase, no added words.
5. Passages, or parts of passages, that start with "Erläuterung:" are explanations of \
the binding text, not requirements themselves. Say so when your answer rests on one.
6. Answer in the language of the question, precisely and briefly, in the register of \
the regulation.
"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
        "quotes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"passage": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["passage", "text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answerable", "answer", "citations", "quotes"],
    "additionalProperties": False,
}


@dataclass
class Answer:
    question: str
    answerable: bool | None             # None if no model decision exists (abstained / error)
    answer: str
    citations: list[int] = field(default_factory=list)
    quotes: list[dict] = field(default_factory=list)
    passage_ids: list[str] = field(default_factory=list)    # chunk id of passage [1], [2], ...
    abstained_by: str | None = None     # "retrieval", "model" or None
    model: str | None = None            # model that actually served the request
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    @property
    def abstained(self) -> bool:
        return self.abstained_by is not None

    def as_dict(self) -> dict:
        return asdict(self)


def format_passages(hits: list[Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        pages = f"S. {c.page_start}" if c.page_start == c.page_end else f"S. {c.page_start}-{c.page_end}"
        label = " > ".join(c.heading_path) if c.heading_path else c.section_id
        blocks.append(f"[{i}] ({c.doc}, {label}, {pages})\n{c.text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, hits: list[Hit]) -> str:
    return f"<passages>\n{format_passages(hits)}\n</passages>\n\nFrage: {question}"


class Generator:
    def __init__(self, model: str = config.GENERATION_MODEL, client=None,
                 max_tokens: int = config.MAX_TOKENS, effort: str | None = config.GENERATION_EFFORT,
                 allow_fallback: bool = config.ALLOW_MODEL_FALLBACK):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.allow_fallback = allow_fallback

    def _create(self, system: str, prompt: str, schema: dict):
        output_config = {"format": {"type": "json_schema", "schema": schema}}
        if self.effort:
            output_config["effort"] = self.effort
        params = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                      messages=[{"role": "user", "content": prompt}], output_config=output_config)
        if self.allow_fallback:
            return self.client.beta.messages.create(
                **params, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        return self.client.messages.create(**params)

    @staticmethod
    def _parse(response) -> tuple[dict | None, str | None]:
        """JSON payload of the response, or an error description."""
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            return None, f"refusal ({category})" if category else "refusal"
        if response.stop_reason == "max_tokens":
            return None, "max_tokens reached; output incomplete"
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            return None, "no text block in response"
        try:
            return json.loads(text), None
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON: {exc}"

    def answer(self, question: str, hits: list[Hit]) -> Answer:
        response = self._create(SYSTEM_PROMPT, build_prompt(question, hits), ANSWER_SCHEMA)
        result = Answer(
            question=question, answerable=None, answer="",
            passage_ids=[h.chunk.chunk_id for h in hits],
            model=getattr(response, "model", self.model),
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        payload, error = self._parse(response)
        if error:
            result.error = error
            return result
        result.answerable = bool(payload["answerable"])
        result.answer = payload["answer"]
        result.citations = [int(c) for c in payload["citations"]]
        result.quotes = [{"passage": int(q["passage"]), "text": q["text"]} for q in payload["quotes"]]
        if not result.answerable:
            result.abstained_by = "model"
        return result


# --- Optional LLM judge ---------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You check whether an answer is supported by the passages it was given. You do not judge \
whether the answer is correct in the real world, only whether every statement in it \
follows from the passages.

Split the answer into its individual claims. A claim is supported if a passage states \
it or it follows directly from what a passage states. List every claim that is not \
supported. Verdict: "supported" if all claims are supported, "unsupported" if the main \
claim is not, "partially_supported" otherwise.
"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["supported", "partially_supported", "unsupported"]},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "unsupported_claims", "reasoning"],
    "additionalProperties": False,
}


class Judge(Generator):
    """Second opinion on groundedness. Secondary to the deterministic quote check:
    a model grading a model is itself a measurement that needs validating."""

    def __init__(self, model: str = config.JUDGE_MODEL, **kwargs):
        super().__init__(model=model, **kwargs)

    def judge(self, question: str, answer: Answer, hits: list[Hit]) -> dict:
        prompt = (f"<passages>\n{format_passages(hits)}\n</passages>\n\n"
                  f"Frage: {question}\n\n<answer>\n{answer.answer}\n</answer>")
        response = self._create(JUDGE_SYSTEM_PROMPT, prompt, JUDGE_SCHEMA)
        payload, error = self._parse(response)
        if error:
            return {"verdict": None, "unsupported_claims": [], "reasoning": "", "error": error,
                    "model": getattr(response, "model", self.model)}
        payload["error"] = None
        payload["model"] = getattr(response, "model", self.model)
        return payload
