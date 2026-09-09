import json
import os
import re
import ast
from typing import Any, Dict

from groq import Groq

from prompts import (
    JOB_ANALYSIS_PROMPT,
    RESUME_ANALYSIS_PROMPT,
    COMPARISON_PROMPT,
    RECOMMENDATIONS_PROMPT,
)


class AnalyzerError(Exception):
    """Custom exception for resume analysis errors."""


class ResumeAnalyzer:
    """
    Main orchestration class.

    Responsibilities:
    - Communicate with Groq
    - Run AI analysis stages
    - Validate JSON
    - Calculate deterministic score
    - Build final result
    """

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL", self.DEFAULT_MODEL)

        if not self.api_key:
            raise AnalyzerError(
                "GROQ_API_KEY is missing. Please add it to your .env file."
            )

        try:
            self.client = Groq(api_key=self.api_key)
        except Exception as error:
            raise AnalyzerError(f"Could not initialize Groq client: {error}")

    # =====================================================
    # Robust JSON extraction
    # =====================================================

    def _extract_json(self, content: str) -> Dict[str, Any]:
        """
        Attempt to extract a valid JSON object from the raw content.
        Handles Markdown, extra text, malformed whitespace, missing braces,
        unquoted keys, trailing commas, and even Python dict-like strings.
        """
        # 1. Remove Markdown code fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        # 2. If the whole string is quoted (like '"..."'), unquote and strip again
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1].strip()

        # 3. Try to find a JSON object between the first { and last }
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            json_str = cleaned[start:end]
            try:
                return json.loads(json_str, strict=False)
            except json.JSONDecodeError:
                pass  # fall through

        # 4. If no braces found, wrap the entire cleaned string with braces
        #    but first, strip any leading/trailing non-JSON characters
        #    (like stray quotes, spaces, newlines)
        wrapped = "{" + cleaned + "}"
        try:
            return json.loads(wrapped, strict=False)
        except json.JSONDecodeError:
            pass

        # 5. Try to repair common JSON issues:
        #    - Remove trailing commas
        #    - Add missing quotes around keys
        #    - Replace single quotes with double quotes
        repaired = cleaned
        # Remove trailing commas before } or ]
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        # Add quotes to unquoted keys (simple alphanumeric and underscore)
        repaired = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired)
        # Replace single quotes with double quotes (but careful with nested quotes)
        # This is a simple approach; might be improved
        repaired = repaired.replace("'", '"')
        # Try to parse again
        try:
            return json.loads(repaired, strict=False)
        except json.JSONDecodeError:
            pass

        # 6. Last resort: use ast.literal_eval if it looks like a Python dict
        try:
            # Remove trailing commas and convert to Python dict literal
            if cleaned.strip().startswith("{") and cleaned.strip().endswith("}"):
                # Use ast.literal_eval which is safer than eval
                result = ast.literal_eval(cleaned)
                if isinstance(result, dict):
                    return result
        except (SyntaxError, ValueError, TypeError):
            pass

        # 7. If all fail, raise a detailed error with the full raw content
        raise AnalyzerError(
            f"Failed to parse JSON. Raw content (full):\n{content}\n"
            "The AI did not return a valid JSON object."
        )

    # =====================================================
    # Groq JSON Call
    # =====================================================

    def _call_ai(self, prompt: str) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional resume analysis engine. "
                            "Return ONLY valid JSON. Do not return Markdown or any explanatory text. "
                            "The JSON must be a single object with no extra whitespace."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
        except Exception as error:
            raise AnalyzerError(f"Groq API request failed: {error}")

        content = response.choices[0].message.content
        if not content:
            raise AnalyzerError("Groq returned an empty response.")

        return self._extract_json(content)

    # =====================================================
    # Main Workflow (unchanged)
    # =====================================================

    def analyze(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        if not resume_text.strip():
            raise AnalyzerError("Resume text is empty.")
        if not job_description.strip():
            raise AnalyzerError("Job description is empty.")

        # Stage 1: Job Description Analysis
        job_prompt = JOB_ANALYSIS_PROMPT.format(job_description=job_description)
        job_analysis = self._call_ai(job_prompt)

        # Stage 2: Resume Analysis
        resume_prompt = RESUME_ANALYSIS_PROMPT.format(resume_text=resume_text)
        resume_analysis = self._call_ai(resume_prompt)

        # Stage 3: Comparison
        comparison_prompt = COMPARISON_PROMPT.format(
            resume_profile=json.dumps(resume_analysis, ensure_ascii=False),
            job_requirements=json.dumps(job_analysis, ensure_ascii=False),
        )
        comparison = self._call_ai(comparison_prompt)

        # Stage 4: Recommendations
        recommendations_prompt = RECOMMENDATIONS_PROMPT.format(
            resume_profile=json.dumps(resume_analysis, ensure_ascii=False),
            job_requirements=json.dumps(job_analysis, ensure_ascii=False),
            comparison=json.dumps(comparison, ensure_ascii=False),
        )
        recommendations = self._call_ai(recommendations_prompt)

        # Stage 5: Score
        score = self._calculate_score(job_analysis, resume_analysis, comparison)

        # Stage 6: Final result
        final_result = self._build_final_result(score, comparison, recommendations)

        return {
            "score": score,
            "job_analysis": job_analysis,
            "resume_analysis": resume_analysis,
            "comparison": comparison,
            "ats": {"keywords": comparison.get("ats_keywords", [])},
            "recommendations": recommendations,
            "final_result": final_result,
        }

    # =====================================================
    # Score Calculation and Final Result (unchanged)
    # =====================================================

    @staticmethod
    def _calculate_score(job, resume, comparison) -> Dict[str, Any]:
        required_skills = {
            str(skill).strip().lower()
            for skill in job.get("required_skills", [])
            if str(skill).strip()
        }
        preferred_skills = {
            str(skill).strip().lower()
            for skill in job.get("preferred_skills", [])
            if str(skill).strip()
        }
        matching_skills = {
            str(skill).strip().lower()
            for skill in comparison.get("matching_skills", [])
            if str(skill).strip()
        }
        partial_matches = {
            str(skill).strip().lower()
            for skill in comparison.get("partial_matches", [])
            if str(skill).strip()
        }

        skill_base = required_skills if required_skills else (required_skills | preferred_skills)
        if skill_base:
            skill_score = round(
                100
                * (len(matching_skills & skill_base) + 0.5 * len(partial_matches & skill_base))
                / len(skill_base)
            )
        else:
            skill_score = 0

        # Other scores
        experience_score = int(comparison.get("experience_match_score", 0) or 0)
        education_score = int(comparison.get("education_match_score", 0) or 0)
        responsibility_score = int(comparison.get("responsibility_match_score", 0) or 0)

        keywords = comparison.get("ats_keywords", [])
        if keywords:
            found_keywords = sum(
                1 for k in keywords if str(k.get("status", "")).lower() == "found"
            )
            keyword_score = round(100 * found_keywords / len(keywords))
        else:
            keyword_score = 0

        breakdown = {
            "skills": max(0, min(100, skill_score)),
            "experience": max(0, min(100, experience_score)),
            "education": max(0, min(100, education_score)),
            "keywords": max(0, min(100, keyword_score)),
            "responsibilities": max(0, min(100, responsibility_score)),
        }

        overall_score = round(
            breakdown["skills"] * 0.30
            + breakdown["experience"] * 0.25
            + breakdown["education"] * 0.15
            + breakdown["keywords"] * 0.15
            + breakdown["responsibilities"] * 0.15
        )
        return {
            "overall": max(0, min(100, overall_score)),
            "keyword_coverage": breakdown["keywords"],
            "breakdown": breakdown,
        }

    @staticmethod
    def _build_final_result(score, comparison, recommendations) -> Dict[str, Any]:
        overall = score["overall"]
        if overall >= 80:
            verdict = "Strong Match"
        elif overall >= 60:
            verdict = "Moderate Match"
        else:
            verdict = "Weak Match"
        return {
            "verdict": verdict,
            "summary": recommendations.get("final_summary", "Review the detailed analysis."),
            "top_strengths": comparison.get("top_strengths", []),
            "top_gaps": comparison.get("top_gaps", []),
        }
