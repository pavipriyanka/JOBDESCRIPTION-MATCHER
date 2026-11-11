"""Optional LLM step: rewrite resume bullets to fit the JD (Gemini).

The prompt forbids inventing skills or experience. It only rephrases what is
already in the resume, so the output stays truthful.
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .config import Settings
from .scoring import MatchReport

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a resume coach for entry-level tech candidates. Improve the candidate's resume for the job "
     "description.\nStrict rules:\n"
     "- NEVER invent skills, tools, employers, numbers or experience that are not in the resume.\n"
     "- Rephrase existing bullets to use the JD's wording where it is truthful.\n"
     "- Use the format: action verb + what was built + tech used + result.\n"
     "- Where a number would help but is missing, write [add number] instead of making one up.\n\n"
     "Return Markdown with two sections:\n"
     "### Rewritten bullets\n(up to 5 bullets: original → improved)\n"
     "### Tailored summary\n(a 2-line professional summary for this role)"),
    ("human",
     "Job description:\n{jd}\n\nResume:\n{resume}\n\n"
     "Required skills the resume is missing (do NOT add these unless the resume shows them): {missing}"),
])


def get_llm(settings: Settings, api_key: str | None = None) -> BaseChatModel | None:
    key = api_key or settings.google_api_key
    if not key:
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=settings.llm_model, google_api_key=key, temperature=0.3, max_retries=2)


def rewrite_suggestions(llm: BaseChatModel, resume_text: str, jd_text: str, report: MatchReport) -> str:
    chain = REWRITE_PROMPT | llm | StrOutputParser()
    return chain.invoke({
        "jd": jd_text[:6000],
        "resume": resume_text[:8000],
        "missing": ", ".join(report.missing_required) or "none",
    })
