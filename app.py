import streamlit as st
from dotenv import load_dotenv

from analyzer import ResumeAnalyzer, AnalyzerError
from resume_parser import ResumeParser, ResumeParserError

# Load environment variables from .env
load_dotenv()

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("📄 AI Resume Analyzer")

st.markdown(
    """
    Compare your resume against a job description using AI.

    Upload your resume, paste the job description, and get:
    **match score, matching skills, missing skills, ATS keywords,
    problems, recommendations, and final result.**
    """
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("About")

    st.write(
        """
        This application analyzes your resume against a specific
        job description.

        The application uses:

        • Python for file processing and scoring  
        • Groq AI for semantic analysis  
        • Streamlit for the interface  
        • PDF/DOCX extraction for resumes
        """
    )

    st.divider()

    st.warning(
        "Never add a skill or experience to your resume unless "
        "you genuinely have it."
    )

# ---------------------------------------------------------
# Input Section
# ---------------------------------------------------------

st.header("1. Provide Your Information")

resume_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx"],
    help="Supported formats: PDF and DOCX",
)

job_description = st.text_area(
    "Paste Job Description",
    height=300,
    placeholder="Paste the complete job description here...",
)

# ---------------------------------------------------------
# Analyze Button
# ---------------------------------------------------------

analyze_button = st.button(
    "🔍 Analyze Resume",
    type="primary",
    use_container_width=True,
)

# ---------------------------------------------------------
# Analysis
# ---------------------------------------------------------

if analyze_button:

    # Validate resume
    if resume_file is None:
        st.error("Please upload your resume.")
        st.stop()

    # Validate job description
    if not job_description.strip():
        st.error("Please paste the job description.")
        st.stop()

    try:

        # -------------------------------------------------
        # Step 1: Extract Resume
        # -------------------------------------------------

        with st.spinner("📄 Extracting resume text..."):

            resume_text = ResumeParser.extract_text(
                file_bytes=resume_file.getvalue(),
                filename=resume_file.name,
            )

        if len(resume_text.strip()) < 100:
            st.error(
                "Very little text could be extracted from the resume. "
                "Please upload a readable PDF/DOCX file."
            )
            st.stop()

        # -------------------------------------------------
        # Step 2: AI Analysis
        # -------------------------------------------------

        with st.spinner(
            "🤖 AI is analyzing your resume and job description..."
        ):

            analyzer = ResumeAnalyzer()

            result = analyzer.analyze(
                resume_text=resume_text,
                job_description=job_description,
            )

        # Store results
        st.session_state["analysis_result"] = result
        st.session_state["resume_text"] = resume_text

        st.success("Analysis completed successfully!")

    except ResumeParserError as error:
        st.error(f"Resume parsing error: {error}")

    except AnalyzerError as error:
        st.error(f"Analysis error: {error}")

    except Exception as error:
        st.error(
            f"Unexpected error occurred: {error}"
        )

# ---------------------------------------------------------
# Display Results
# ---------------------------------------------------------

result = st.session_state.get("analysis_result")

if result:

    st.divider()

    st.header("2. Resume Analysis Results")

    # -----------------------------------------------------
    # Score Section
    # -----------------------------------------------------

    score = result["score"]

    overall_score = score["overall"]

    verdict = result["final_result"]["verdict"]

    keyword_coverage = score["keyword_coverage"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Overall Match",
            f"{overall_score}/100",
        )

    with col2:
        st.metric(
            "Final Verdict",
            verdict,
        )

    with col3:
        st.metric(
            "ATS Keyword Coverage",
            f"{keyword_coverage}%",
        )

    st.progress(overall_score / 100)

    # -----------------------------------------------------
    # Final Summary
    # -----------------------------------------------------

    st.subheader("📋 Final Result")

    st.info(
        result["final_result"]["summary"]
    )

    # -----------------------------------------------------
    # Matching / Missing Skills
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    comparison = result["comparison"]

    with col1:

        st.subheader("✅ Matching Skills")

        matching_skills = comparison.get(
            "matching_skills",
            [],
        )

        if matching_skills:

            for skill in matching_skills:
                st.write(f"✓ {skill}")

        else:
            st.write("No strong matching skills identified.")

    with col2:

        st.subheader("❌ Missing Skills")

        missing_skills = comparison.get(
            "missing_skills",
            [],
        )

        if missing_skills:

            for skill in missing_skills:
                st.write(f"✗ {skill}")

        else:
            st.write("No major missing skills identified.")

    # -----------------------------------------------------
    # Partial Matches
    # -----------------------------------------------------

    st.subheader("🟡 Partial Matches")

    partial_matches = comparison.get(
        "partial_matches",
        [],
    )

    if partial_matches:

        for skill in partial_matches:
            st.write(f"• {skill}")

    else:
        st.write("No partial matches identified.")

    # -----------------------------------------------------
    # ATS Keywords
    # -----------------------------------------------------

    st.subheader("🔑 ATS Keywords")

    ats_keywords = result["ats"]["keywords"]

    if ats_keywords:

        keyword_rows = []

        for item in ats_keywords:

            keyword_rows.append(
                {
                    "Keyword": item.get(
                        "keyword",
                        "",
                    ),
                    "Priority": item.get(
                        "priority",
                        "",
                    ),
                    "Status": item.get(
                        "status",
                        "",
                    ),
                }
            )

        st.dataframe(
            keyword_rows,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.write(
            "No ATS keywords were identified."
        )

    # -----------------------------------------------------
    # Score Breakdown
    # -----------------------------------------------------

    st.subheader("📊 Score Breakdown")

    breakdown = score["breakdown"]

    score_rows = []

    for category, value in breakdown.items():

        score_rows.append(
            {
                "Category": category.replace(
                    "_",
                    " ",
                ).title(),
                "Score": value,
            }
        )

    st.dataframe(
        score_rows,
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------
    # Problems
    # -----------------------------------------------------

    st.subheader("⚠️ Resume Problems")

    problems = result["recommendations"].get(
        "problems",
        [],
    )

    if problems:

        for problem in problems:
            st.write(f"• {problem}")

    else:

        st.write(
            "No major problems were identified."
        )

    # -----------------------------------------------------
    # Recommendations
    # -----------------------------------------------------

    st.subheader("💡 Recommendations")

    recommendations = result[
        "recommendations"
    ].get(
        "recommendations",
        [],
    )

    if recommendations:

        for recommendation in recommendations:
            st.write(
                f"• {recommendation}"
            )

    else:

        st.write(
            "No recommendations generated."
        )

    # -----------------------------------------------------
    # Strengths
    # -----------------------------------------------------

    st.subheader("💪 Top Strengths")

    strengths = comparison.get(
        "top_strengths",
        [],
    )

    if strengths:

        for strength in strengths:
            st.write(f"✓ {strength}")

    # -----------------------------------------------------
    # Gaps
    # -----------------------------------------------------

    st.subheader("🎯 Top Gaps")

    gaps = comparison.get(
        "top_gaps",
        [],
    )

    if gaps:

        for gap in gaps:
            st.write(f"• {gap}")

    # -----------------------------------------------------
    # Evidence Notes
    # -----------------------------------------------------

    with st.expander(
        "🔎 View AI Evidence Notes"
    ):

        evidence_notes = comparison.get(
            "evidence_notes",
            [],
        )

        for note in evidence_notes:
            st.write(f"• {note}")

    # -----------------------------------------------------
    # Extracted Resume
    # -----------------------------------------------------

    with st.expander(
        "📄 View Extracted Resume Text"
    ):

        resume_text = st.session_state.get(
            "resume_text",
            "",
        )

        st.text(resume_text)

    # -----------------------------------------------------
    # Structured JSON
    # -----------------------------------------------------

    with st.expander(
        "🧩 View Complete Structured JSON"
    ):

        st.json(result)

    # -----------------------------------------------------
    # Disclaimer
    # -----------------------------------------------------

    st.divider()

    st.caption(
        "This tool provides AI-assisted resume analysis. "
        "It does not guarantee ATS behavior or job selection. "
        "Always verify recommendations and only include "
        "skills and experience that you genuinely possess."
    )