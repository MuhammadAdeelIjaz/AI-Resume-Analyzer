import json
import streamlit as st
from dotenv import load_dotenv

from resume_parser import ResumeParser, ResumeParserError
from analyzer import ResumeAnalyzer, AnalyzerError

load_dotenv()

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Resume Analyzer")
st.write("Compare your resume against a job description using AI.")
st.write(
    "Upload your resume, paste the job description, and get: "
    "**match score, matching skills, missing skills, ATS keywords, "
    "problems, recommendations, and final result.**"
)

st.header("1. Provide Your Information")

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx"],
    help="Supported formats: PDF and DOCX",
)

job_description = st.text_area(
    "Paste Job Description",
    height=220,
    placeholder="Paste the complete job description here...",
)

analyze_clicked = st.button(
    "🔍 Analyze Resume",
    type="primary",
    use_container_width=True,
)

if analyze_clicked:
    if uploaded_file is None:
        st.error("Please upload a PDF or DOCX resume.")
        st.stop()

    if not job_description.strip():
        st.error("Please paste a job description.")
        st.stop()

    try:
        with st.spinner("Extracting resume text..."):
            resume_text = ResumeParser.extract_text(
                uploaded_file.getvalue(),
                uploaded_file.name,
            )

        if len(resume_text.strip()) < 100:
            st.error(
                "Very little readable text was found in the resume. "
                "If this is a scanned/image-only PDF, OCR may be required."
            )
            st.stop()

        with st.spinner("Analyzing resume and job description..."):
            analyzer = ResumeAnalyzer()
            result = analyzer.analyze(
                resume_text=resume_text,
                job_description=job_description,
            )

        st.session_state["analysis_result"] = result
        st.session_state["resume_text"] = resume_text

        st.success("Analysis completed successfully.")

    except ResumeParserError as exc:
        st.error(f"Resume parsing error: {exc}")
    except AnalyzerError as exc:
        st.error(f"AI analysis error: {exc}")
    except Exception as exc:
        st.error(f"Unexpected error occurred: {exc}")

result = st.session_state.get("analysis_result")

if result:
    st.divider()
    st.header("2. Analysis Result")

    score = result.get("overall_score", 0)
    verdict = result.get("verdict", "Unknown")
    keyword_coverage = result.get("ats_keyword_coverage", 0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Match Score", f"{score}/100")
    col2.metric("Verdict", verdict)
    col3.metric("ATS Keyword Coverage", f"{keyword_coverage}%")

    st.subheader("Summary")
    st.write(result.get("final_summary", ""))

    st.subheader("Matching Skills")
    matching = result.get("matching_skills", [])
    st.write(", ".join(matching) if matching else "No strong matching skills identified.")

    st.subheader("Missing Skills")
    missing = result.get("missing_skills", [])
    st.write(", ".join(missing) if missing else "No major missing skills identified.")

    st.subheader("Partial Matches")
    partial = result.get("partial_matches", [])
    if partial:
        for item in partial:
            if isinstance(item, dict):
                st.write(
                    f"**{item.get('skill', 'Unknown')}**: "
                    f"{item.get('reason', '')}"
                )
            else:
                st.write(str(item))
    else:
        st.write("No partial matches identified.")

    st.subheader("ATS Keywords")
    keywords = result.get("ats_keywords", [])
    if keywords:
        st.dataframe(
            keywords,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No ATS keywords returned.")

    st.subheader("Score Breakdown")
    breakdown = result.get("score_breakdown", {})
    if breakdown:
        for name, value in breakdown.items():
            st.write(f"**{name.replace('_', ' ').title()}:** {value}/100")

    st.subheader("Problems")
    problems = result.get("problems", [])
    if problems:
        for item in problems:
            st.warning(str(item))
    else:
        st.write("No major problems identified.")

    st.subheader("Recommendations")
    recommendations = result.get("recommendations", [])
    if recommendations:
        for item in recommendations:
            st.info(str(item))
    else:
        st.write("No recommendations returned.")

    st.subheader("Top Strengths")
    for item in result.get("top_strengths", []):
        st.write(f"✅ {item}")

    st.subheader("Top Gaps")
    for item in result.get("top_gaps", []):
        st.write(f"⚠️ {item}")

    st.subheader("Evidence Notes")
    evidence = result.get("evidence_notes", [])
    for item in evidence:
        st.write(f"• {item}")

    with st.expander("Extracted Resume Text"):
        st.text(st.session_state.get("resume_text", ""))

    with st.expander("Complete Structured JSON"):
        st.json(result)

    st.caption(
        "Important: Recommendations should improve truthful presentation of "
        "your existing qualifications. Do not fabricate skills, experience, "
        "certifications, projects, or achievements."
    )
