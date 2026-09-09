import json
import os
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

        self.api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
        )

        self.model = os.getenv(
            "GROQ_MODEL",
            self.DEFAULT_MODEL,
        )

        if not self.api_key:

            raise AnalyzerError(
                "GROQ_API_KEY is missing. "
                "Please add it to your .env file."
            )

        try:

            self.client = Groq(
                api_key=self.api_key
            )

        except Exception as error:

            raise AnalyzerError(
                f"Could not initialize Groq client: {error}"
            )

    # =====================================================
    # Groq JSON Call
    # =====================================================

    def _call_ai(
        self,
        prompt: str,
    ) -> Dict[str, Any]:

        try:

            response = (
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a professional resume "
                                "analysis engine. "
                                "Return ONLY valid JSON. "
                                "Do not return Markdown."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=0.1,
                    response_format={
                        "type": "json_object"
                    },
                )
            )

        except Exception as error:

            raise AnalyzerError(
                f"Groq API request failed: {error}"
            )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:

            raise AnalyzerError(
                "Groq returned an empty response."
            )

        try:

            return json.loads(content)

        except json.JSONDecodeError:

            raise AnalyzerError(
                "Groq returned invalid JSON."
            )

    # =====================================================
    # Main Workflow
    # =====================================================

    def analyze(
        self,
        resume_text: str,
        job_description: str,
    ) -> Dict[str, Any]:

        if not resume_text.strip():

            raise AnalyzerError(
                "Resume text is empty."
            )

        if not job_description.strip():

            raise AnalyzerError(
                "Job description is empty."
            )

        # -------------------------------------------------
        # Stage 1: Job Description Analysis
        # -------------------------------------------------

        job_prompt = (
            JOB_ANALYSIS_PROMPT
            .format(
                job_description=job_description
            )
        )

        job_analysis = self._call_ai(
            job_prompt
        )

        # -------------------------------------------------
        # Stage 2: Resume Analysis
        # -------------------------------------------------

        resume_prompt = (
            RESUME_ANALYSIS_PROMPT
            .format(
                resume_text=resume_text
            )
        )

        resume_analysis = self._call_ai(
            resume_prompt
        )

        # -------------------------------------------------
        # Stage 3: Comparison
        # -------------------------------------------------

        comparison_prompt = (
            COMPARISON_PROMPT
            .format(
                resume_profile=json.dumps(
                    resume_analysis,
                    ensure_ascii=False,
                ),
                job_requirements=json.dumps(
                    job_analysis,
                    ensure_ascii=False,
                ),
            )
        )

        comparison = self._call_ai(
            comparison_prompt
        )

        # -------------------------------------------------
        # Stage 4: Recommendations
        # -------------------------------------------------

        recommendations_prompt = (
            RECOMMENDATIONS_PROMPT
            .format(
                resume_profile=json.dumps(
                    resume_analysis,
                    ensure_ascii=False,
                ),
                job_requirements=json.dumps(
                    job_analysis,
                    ensure_ascii=False,
                ),
                comparison=json.dumps(
                    comparison,
                    ensure_ascii=False,
                ),
            )
        )

        recommendations = self._call_ai(
            recommendations_prompt
        )

        # -------------------------------------------------
        # Stage 5: Deterministic Score
        # -------------------------------------------------

        score = self._calculate_score(
            job_analysis,
            resume_analysis,
            comparison,
        )

        # -------------------------------------------------
        # Stage 6: Final Result
        # -------------------------------------------------

        final_result = (
            self._build_final_result(
                score,
                comparison,
                recommendations,
            )
        )

        return {
            "score": score,
            "job_analysis": job_analysis,
            "resume_analysis": resume_analysis,
            "comparison": comparison,
            "ats": {
                "keywords": comparison.get(
                    "ats_keywords",
                    [],
                )
            },
            "recommendations": recommendations,
            "final_result": final_result,
        }

    # =====================================================
    # Score Calculation
    # =====================================================

    @staticmethod
    def _calculate_score(
        job: Dict[str, Any],
        resume: Dict[str, Any],
        comparison: Dict[str, Any],
    ) -> Dict[str, Any]:

        # -------------------------------------------------
        # Required Skills
        # -------------------------------------------------

        required_skills = {
            str(skill)
            .strip()
            .lower()
            for skill in job.get(
                "required_skills",
                [],
            )
            if str(skill).strip()
        }

        preferred_skills = {
            str(skill)
            .strip()
            .lower()
            for skill in job.get(
                "preferred_skills",
                [],
            )
            if str(skill).strip()
        }

        # -------------------------------------------------
        # Matching Skills
        # -------------------------------------------------

        matching_skills = {
            str(skill)
            .strip()
            .lower()
            for skill in comparison.get(
                "matching_skills",
                [],
            )
            if str(skill).strip()
        }

        partial_matches = {
            str(skill)
            .strip()
            .lower()
            for skill in comparison.get(
                "partial_matches",
                [],
            )
            if str(skill).strip()
        }

        # -------------------------------------------------
        # Calculate Skill Score
        # -------------------------------------------------

        skill_base = required_skills

        if not skill_base:

            skill_base = (
                required_skills
                | preferred_skills
            )

        if skill_base:

            skill_score = round(
                100
                * (
                    len(
                        matching_skills
                        & skill_base
                    )
                    + (
                        0.5
                        * len(
                            partial_matches
                            & skill_base
                        )
                    )
                )
                / len(skill_base)
            )

        else:

            skill_score = 0

        # -------------------------------------------------
        # Other AI Scores
        # -------------------------------------------------

        experience_score = (
            comparison.get(
                "experience_match_score",
                0,
            )
        )

        education_score = (
            comparison.get(
                "education_match_score",
                0,
            )
        )

        responsibility_score = (
            comparison.get(
                "responsibility_match_score",
                0,
            )
        )

        # Ensure values are integers
        try:
            experience_score = int(
                experience_score
            )
        except (TypeError, ValueError):
            experience_score = 0

        try:
            education_score = int(
                education_score
            )
        except (TypeError, ValueError):
            education_score = 0

        try:
            responsibility_score = int(
                responsibility_score
            )
        except (TypeError, ValueError):
            responsibility_score = 0

        # -------------------------------------------------
        # ATS Keyword Score
        # -------------------------------------------------

        keywords = comparison.get(
            "ats_keywords",
            [],
        )

        if keywords:

            found_keywords = sum(
                1
                for keyword in keywords
                if str(
                    keyword.get(
                        "status",
                        ""
                    )
                ).lower()
                == "found"
            )

            keyword_score = round(
                100
                * found_keywords
                / len(keywords)
            )

        else:

            keyword_score = 0

        # -------------------------------------------------
        # Clamp Values
        # -------------------------------------------------

        breakdown = {

            "skills": max(
                0,
                min(100, skill_score),
            ),

            "experience": max(
                0,
                min(100, experience_score),
            ),

            "education": max(
                0,
                min(100, education_score),
            ),

            "keywords": max(
                0,
                min(100, keyword_score),
            ),

            "responsibilities": max(
                0,
                min(100, responsibility_score),
            ),
        }

        # -------------------------------------------------
        # Weighted Score
        # -------------------------------------------------

        overall_score = round(

            breakdown["skills"] * 0.30

            + breakdown["experience"] * 0.25

            + breakdown["education"] * 0.15

            + breakdown["keywords"] * 0.15

            + breakdown["responsibilities"] * 0.15
        )

        return {

            "overall": max(
                0,
                min(100, overall_score),
            ),

            "keyword_coverage": breakdown[
                "keywords"
            ],

            "breakdown": breakdown,
        }

    # =====================================================
    # Final Result
    # =====================================================

    @staticmethod
    def _build_final_result(
        score: Dict[str, Any],
        comparison: Dict[str, Any],
        recommendations: Dict[str, Any],
    ) -> Dict[str, Any]:

        overall = score["overall"]

        if overall >= 80:

            verdict = "Strong Match"

        elif overall >= 60:

            verdict = "Moderate Match"

        else:

            verdict = "Weak Match"

        return {

            "verdict": verdict,

            "summary": recommendations.get(
                "final_summary",
                "Review the detailed analysis.",
            ),

            "top_strengths": comparison.get(
                "top_strengths",
                [],
            ),

            "top_gaps": comparison.get(
                "top_gaps",
                [],
            ),
        }