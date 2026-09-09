# =========================================================
# JOB DESCRIPTION ANALYSIS
# =========================================================

JOB_ANALYSIS_PROMPT = """
Analyze the following job description.

Your task is to identify the actual requirements of the
job.

IMPORTANT RULES:

1. Extract only information supported by the job description.
2. Do not invent requirements.
3. Separate required skills from preferred skills.
4. Identify experience requirements.
5. Identify education requirements.
6. Identify certifications.
7. Identify major responsibilities.
8. Identify important ATS keywords.
9. Keep skills and keywords concise.
10. Return ONLY valid JSON.

Return JSON using exactly this structure:

{
    "job_title": "",
    "required_skills": [],
    "preferred_skills": [],
    "required_experience": [],
    "education_requirements": [],
    "certifications": [],
    "responsibilities": [],
    "important_keywords": []
}

JOB DESCRIPTION:

{job_description}
"""


# =========================================================
# RESUME ANALYSIS
# =========================================================

RESUME_ANALYSIS_PROMPT = """
Analyze the following resume.

Extract the candidate's actual qualifications.

IMPORTANT RULES:

1. Use only information explicitly present in the resume.
2. Do not invent skills.
3. Do not infer skills merely because they are common
   for the candidate's profession.
4. Do not invent experience.
5. Do not invent certifications.
6. Identify technologies explicitly mentioned.
7. Identify projects explicitly mentioned.
8. Preserve important evidence from work experience.
9. Return ONLY valid JSON.

Return JSON using exactly this structure:

{
    "name": "",
    "professional_summary": "",
    "skills": [],
    "experience": [
        {
            "role": "",
            "company": "",
            "duration": "",
            "highlights": []
        }
    ],
    "education": [],
    "certifications": [],
    "projects": [],
    "technologies": []
}

RESUME:

{resume_text}
"""


# =========================================================
# RESUME / JOB COMPARISON
# =========================================================

COMPARISON_PROMPT = """
Compare the candidate profile with the job requirements.

Your goal is to determine how well the candidate's actual
qualifications align with the job.

IMPORTANT RULES:

1. A skill is a MATCH only when the resume provides
   credible evidence for it.

2. A skill is PARTIAL when the resume shows related,
   incomplete, or indirect evidence.

3. A skill is MISSING when the job requires it and the
   resume does not provide credible evidence.

4. Do not claim the candidate has a skill that is not
   supported by the resume.

5. Consider semantic similarity, not only exact wording.

6. However, do not treat unrelated technologies as matches.

7. Experience scores must be integers from 0 to 100.

8. Education score must be from 0 to 100.

9. Responsibility score must be from 0 to 100.

10. ATS keyword status must be one of:
    "Found"
    "Partial"
    "Missing"

11. ATS priority must be one of:
    "High"
    "Medium"
    "Low"

12. Include only meaningful job-specific keywords.

13. Return ONLY valid JSON.

Return JSON using exactly this structure:

{
    "matching_skills": [],
    "partial_matches": [],
    "missing_skills": [],

    "experience_match_score": 0,

    "education_match_score": 0,

    "responsibility_match_score": 0,

    "top_strengths": [],

    "top_gaps": [],

    "ats_keywords": [
        {
            "keyword": "",
            "priority": "High",
            "status": "Found"
        }
    ],

    "evidence_notes": []
}

CANDIDATE PROFILE:

{resume_profile}

JOB REQUIREMENTS:

{job_requirements}
"""


# =========================================================
# RECOMMENDATIONS
# =========================================================

RECOMMENDATIONS_PROMPT = """
Generate practical resume improvement recommendations.

Use the candidate profile, job requirements, and comparison.

IMPORTANT RULES:

1. Never tell the candidate to fabricate information.

2. Never recommend falsely adding:
   - skills
   - certifications
   - experience
   - projects
   - technologies
   - achievements

3. If a requirement is genuinely missing, describe it as
   a skill gap.

4. If the candidate already has relevant experience but
   the resume does not present it clearly, recommend
   improving its presentation.

5. Recommendations should be specific.

6. Focus on changes that could improve alignment with
   this particular job.

7. Identify problems in the current resume.

8. Keep recommendations concise.

9. Return ONLY valid JSON.

Return JSON using exactly this structure:

{
    "problems": [],
    "recommendations": [],
    "final_summary": ""
}

CANDIDATE PROFILE:

{resume_profile}

JOB REQUIREMENTS:

{job_requirements}

COMPARISON:

{comparison}
"""