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

    DEFAULT_MODEL = "mixtral-8x7b-32768"

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
    # Ultra‑robust JSON extraction
    # =====================================================

    def _extract_json(self, content: str) -> Dict[str, Any]:
        """
        Attempt to extract a valid JSON object from the raw content.
        Tries multiple strategies in order.
        """
        raw = content.strip()

        # Remove Markdown code fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

        if not raw:
            raise AnalyzerError("Empty response from AI.")

        # Helper to try parsing
        def try_parse(s):
            try:
                return json.loads(s, strict=False)
            except:
                return None

        # ----- 1. Direct parse -----
        result = try_parse(raw)
        if result is not None:
            return result

        # ----- 2. If whole string is quoted, unquote and try -----
        if raw.startswith('"') and raw.endswith('"'):
            unquoted = raw[1:-1].strip()
            result = try_parse(unquoted)
            if result is not None:
                return result

        # ----- 3. Extract between first { and last } -----
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            json_str = raw[start:end]
            result = try_parse(json_str)
            if result is not None:
                return result

        # ----- 4. Strip leading/trailing junk and wrap with braces -----
        cleaned = re.sub(r'^[\s"\'`]+', '', raw)
        cleaned = re.sub(r'[\s"\'`]+$', '', cleaned)
        wrapped = "{" + cleaned + "}"
        result = try_parse(wrapped)
        if result is not None:
            return result

        # ----- 5. Repair common issues: trailing commas, unquoted keys, single quotes -----
        repaired = cleaned
        # Remove trailing commas before } or ]
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
        # Quote unquoted keys
        repaired = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired)
        # Replace single quotes with double quotes
        repaired = repaired.replace("'", '"')
        wrapped = "{" + repaired + "}"
        result = try_parse(wrapped)
        if result is not None:
            return result

        # ----- 6. Use ast.literal_eval if it looks like a Python dict -----
        try:
            if cleaned.strip().startswith("{") and cleaned.strip().endswith("}"):
                result = ast.literal_eval(cleaned)
                if isinstance(result, dict):
                    return result
        except (SyntaxError, ValueError, TypeError):
            pass

        # ----- 7. Manually extract key-value pairs (even without braces) -----
        try:
            # Find all patterns like "key": value (value may be string, number, array, object)
            # We'll capture until a comma or end, but handle nested structures poorly.
            # Better: use regex that matches keys and values, but we'll do a simple version.
            # Since the error shows "job_title", we can try to parse a simple object.
            # We'll attempt to find all quoted keys and their values.
            pattern = r'"([^"]+)"\s*:\s*([^,]+)(?=,|$)'
            matches = re.findall(pattern, raw)
            if matches:
                result_dict = {}
                for key, value in matches:
                    v = value.strip()
                    parsed = try_parse(v)
                    if parsed is None:
                        # If it's a string without quotes, add them
                        if not (v.startswith('"') and v.endswith('"')):
                            v = '"' + v + '"'
                        parsed = try_parse(v)
                    if parsed is None:
                        parsed = v
                    result_dict[key] = parsed
                return result_dict
        except Exception:
            pass

        # ----- 8. All failed: raise error with full content -----
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
