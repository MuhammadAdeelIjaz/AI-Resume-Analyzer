"""
Prompt templates for the AI Resume Analyzer.

Important:
These prompts intentionally avoid Python .format() placeholders because
the prompts contain JSON braces. Using .format() on JSON examples can cause:

KeyError: '\n "job_title"'

All dynamic values are inserted through simple string concatenation.
"""


def build_job_analysis_prompt(job_description: str) -> str:
    return """You are an expert job-description analyzer.

Analyze the following job description and return ONLY valid JSON.

Required JSON structure:
{
  "job_title": "string",
  "required_skills": ["skill1", "skill2"],
  "preferred_skills": ["skill1", "skill2"],
  "required_experience": "string",
  "education_requirements": ["requirement1"],
  "certifications": ["certification1"],
  "responsibilities": ["responsibility1"],
  "important_keywords": ["keyword1", "keyword2"]
}

Rules:
- Extract requirements explicitly stated or strongly implied by the job description.
- Keep skills concise and specific.
- Do not invent requirements.
- Return JSON only. No markdown and no explanation.

JOB DESCRIPTION:
""" + job_description


def build_resume_analysis_prompt(resume_text: str) -> str:
    return """You are an expert resume parser.

Analyze the following resume and return ONLY valid JSON.

Required JSON structure:
{
  "name": "string",
  "professional_summary": "string",
  "skills": ["skill1", "skill2"],
  "experience": [
    {
      "role": "string",
      "company": "string",
      "duration": "string",
      "highlights": ["highlight1", "highlight2"]
    }
  ],
  "education": [
    {
      "degree": "string",
      "institution": "string",
      "year": "string"
    }
  ],
  "certifications": ["certification1"],
  "projects": ["project1"],
  "technologies": ["technology1", "technology2"]
}

Rules:
- Extract only information supported by the resume.
- Do not invent skills, experience, education, projects, or certifications.
- Return JSON only. No markdown and no explanation.

RESUME:
""" + resume_text


def build_comparison_prompt(job_analysis: dict, resume_analysis: dict) -> str:
    import json

    job_json = json.dumps(job_analysis, ensure_ascii=False, indent=2)
    resume_json = json.dumps(resume_analysis, ensure_ascii=False, indent=2)

    return """You are an expert ATS and recruitment analyst.

Compare the job requirements with the resume information below.

Return ONLY valid JSON using exactly this structure:
{
  "matching_skills": ["skill1", "skill2"],
  "partial_matches": [
    {
      "skill": "skill",
      "reason": "short explanation"
    }
  ],
  "missing_skills": ["skill1", "skill2"],
  "experience_match_score": 0,
  "education_match_score": 0,
  "responsibility_match_score": 0,
  "top_strengths": ["strength1"],
  "top_gaps": ["gap1"],
  "ats_keywords": [
    {
      "keyword": "keyword",
      "priority": "High",
      "status": "Found"
    }
  ],
  "evidence_notes": ["short evidence note"]
}

Scoring rules:
- experience_match_score: 0 to 100
- education_match_score: 0 to 100
- responsibility_match_score: 0 to 100
- ATS keyword status must be exactly one of: Found, Partial, Missing
- ATS priority must be exactly one of: High, Medium, Low
- A skill is "matching" only when the resume clearly demonstrates it.
- Use "Partial" when the resume shows related or incomplete evidence.
- Do not treat unrelated technologies as matches.
- Do not invent evidence.
- Return JSON only.

JOB ANALYSIS:
""" + job_json + """

RESUME ANALYSIS:
""" + resume_json


def build_recommendations_prompt(
    job_analysis: dict,
    resume_analysis: dict,
    comparison: dict,
) -> str:
    import json

    job_json = json.dumps(job_analysis, ensure_ascii=False, indent=2)
    resume_json = json.dumps(resume_analysis, ensure_ascii=False, indent=2)
    comparison_json = json.dumps(comparison, ensure_ascii=False, indent=2)

    return """You are an expert resume and ATS improvement advisor.

Based on the job analysis, resume analysis, and comparison, return ONLY valid JSON.

Required JSON structure:
{
  "problems": ["problem1", "problem2"],
  "recommendations": ["recommendation1", "recommendation2"],
  "final_summary": "short overall assessment"
}

Rules:
- Recommendations must be truthful.
- Never tell the candidate to fabricate skills, experience, certifications,
  projects, education, technologies, or achievements.
- Recommend adding a skill only when phrased as a future learning or development
  suggestion, not as something to claim immediately.
- Focus on ATS keywords, clarity, measurable achievements, relevance,
  structure, and evidence.
- Return JSON only.

JOB ANALYSIS:
""" + job_json + """

RESUME ANALYSIS:
""" + resume_json + """

COMPARISON:
""" + comparison_json
