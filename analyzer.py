import json
import os
from typing import Any

from groq import Groq

from prompts import (
    build_job_analysis_prompt,
    build_resume_analysis_prompt,
    build_comparison_prompt,
    build_recommendations_prompt,
)


class AnalyzerError(Exception):
    """Raised when AI analysis fails."""


class ResumeAnalyzer:
    """Orchestrates AI analysis and deterministic score calculation."""

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self) -> None:
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL", self.DEFAULT_MODEL)

        if not self.api_key:
            raise AnalyzerError(
                "GROQ_API_KEY was not found. "
                "For local use, add it to .env. "
                "For Streamlit Cloud, add it to App Settings > Secrets."
            )

        try:
            self.client = Groq(api_key=self.api_key)
        except Exception as exc:
            raise AnalyzerError(
                f"Could not initialize Groq client: {exc}"
            ) from exc

    def _call_ai(self, prompt: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise resume analysis assistant. "
                            "Return only valid JSON. Do not use markdown fences."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content

            if not content:
                raise AnalyzerError("Groq returned an empty response.")

            return self._parse_json(content)

        except AnalyzerError:
            raise
        except Exception as exc:
            raise AnalyzerError(
                f"Groq API request failed: {exc}"
            ) from exc

    @staticmethod
    def _parse_json(content: str) -> dict:
        cleaned = content.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```json", "", 1)
            cleaned = cleaned.replace("```", "")
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise AnalyzerError(
                f"AI returned invalid JSON: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise AnalyzerError("AI response was not a JSON object.")

        return data

    def analyze(
        self,
        resume_text: str,
        job_description: str,
    ) -> dict:
        if not resume_text.strip():
            raise AnalyzerError("Resume text is empty.")

        if not job_description.strip():
            raise AnalyzerError("Job description is empty.")

        try:
            job_analysis = self._call_ai(
                build_job_analysis_prompt(job_description)
            )

            resume_analysis = self._call_ai(
                build_resume_analysis_prompt(resume_text)
            )

            comparison = self._call_ai(
                build_comparison_prompt(
                    job_analysis,
                    resume_analysis,
                )
            )

            recommendations = self._call_ai(
                build_recommendations_prompt(
                    job_analysis,
                    resume_analysis,
                    comparison,
                )
            )

            score_data = self._calculate_score(
                job_analysis,
                comparison,
            )

            return {
                **comparison,
                **recommendations,
                **score_data,
            }

        except AnalyzerError:
            raise
        except Exception as exc:
            raise AnalyzerError(
                f"Unexpected analysis failure: {exc}"
            ) from exc

    @staticmethod
    def _as_list(value: Any) -> list:
        return value if isinstance(value, list) else []

    @staticmethod
    def _safe_score(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
            return max(0.0, min(100.0, number))
        except (TypeError, ValueError):
            return default

    def _calculate_score(
        self,
        job_analysis: dict,
        comparison: dict,
    ) -> dict:
        required_skills = self._as_list(
            job_analysis.get("required_skills")
        )

        preferred_skills = self._as_list(
            job_analysis.get("preferred_skills")
        )

        matching_skills = self._as_list(
            comparison.get("matching_skills")
        )

        partial_matches = self._as_list(
            comparison.get("partial_matches")
        )

        missing_skills = self._as_list(
            comparison.get("missing_skills")
        )

        # Normalize skill names for fair counting.
        required_normalized = {
            str(skill).strip().lower()
            for skill in required_skills
            if str(skill).strip()
        }

        matching_normalized = {
            str(skill).strip().lower()
            for skill in matching_skills
            if str(skill).strip()
        }

        partial_names = set()

        for item in partial_matches:
            if isinstance(item, dict):
                skill = item.get("skill", "")
            else:
                skill = item

            if str(skill).strip():
                partial_names.add(str(skill).strip().lower())

        if required_normalized:
            matched_required = len(
                required_normalized & matching_normalized
            )
            partial_required = len(
                required_normalized & partial_names
            )
            skill_score = (
                (matched_required + 0.5 * partial_required)
                / len(required_normalized)
            ) * 100
        else:
            skill_score = 0.0

        experience_score = self._safe_score(
            comparison.get("experience_match_score")
        )

        education_score = self._safe_score(
            comparison.get("education_match_score")
        )

        responsibility_score = self._safe_score(
            comparison.get("responsibility_match_score")
        )

        ats_keywords = self._as_list(
            comparison.get("ats_keywords")
        )

        keyword_total = 0.0
        keyword_possible = 0.0

        for item in ats_keywords:
            if not isinstance(item, dict):
                continue

            status = str(item.get("status", "")).strip().lower()
            priority = str(item.get("priority", "")).strip().lower()

            weight = {
                "high": 3.0,
                "medium": 2.0,
                "low": 1.0,
            }.get(priority, 1.0)

            keyword_possible += weight

            if status == "found":
                keyword_total += weight
            elif status == "partial":
                keyword_total += 0.5 * weight

        if keyword_possible:
            keyword_score = (
                keyword_total / keyword_possible
            ) * 100
        else:
            keyword_score = 0.0

        overall_score = round(
            (
                skill_score * 0.30
                + experience_score * 0.25
                + education_score * 0.15
                + keyword_score * 0.15
                + responsibility_score * 0.15
            ),
            1,
        )

        if overall_score >= 80:
            verdict = "Strong Match"
        elif overall_score >= 60:
            verdict = "Moderate Match"
        else:
            verdict = "Weak Match"

        ats_coverage = round(keyword_score, 1)

        return {
            "overall_score": overall_score,
            "verdict": verdict,
            "ats_keyword_coverage": ats_coverage,
            "score_breakdown": {
                "skills": round(skill_score, 1),
                "experience": round(experience_score, 1),
                "education": round(education_score, 1),
                "ats_keywords": round(keyword_score, 1),
                "responsibilities": round(responsibility_score, 1),
            },
            "missing_skill_count": len(missing_skills),
            "preferred_skill_count": len(preferred_skills),
        }
