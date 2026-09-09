# AI Resume Analyzer

A Streamlit application that compares a resume with a job description using the Groq API.

## Features

- PDF and DOCX resume upload
- Job description analysis
- Resume information extraction
- Matching skills
- Missing skills
- Partial matches
- ATS keyword analysis
- Deterministic weighted match score
- Problems and recommendations
- Final verdict

## Architecture

```text
app.py
  |
  +--> resume_parser.py
  |
  +--> analyzer.py
          |
          +--> prompts.py
          |
          +--> Groq API
```

## Important fix

The prompt templates do NOT use Python `.format()` with JSON examples.

Using `.format()` on a prompt containing JSON such as:

```python
{
  "job_title": "string"
}
```

can cause:

```text
KeyError: '\n "job_title"'
```

The corrected project uses prompt-builder functions and string concatenation instead.

## Local setup

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env`:

```text
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

Run:

```bash
streamlit run app.py
```

## GitHub

Upload the source files, but NEVER upload `.env`.

`.env.example` is safe to upload because it contains a placeholder instead of the real API key.

## Streamlit Community Cloud

1. Push the project to GitHub.
2. Create a new app on Streamlit Community Cloud.
3. Select the GitHub repository.
4. Select `app.py` as the main file.
5. Open Advanced settings.
6. Add the following to Secrets:

```toml
GROQ_API_KEY = "your_actual_groq_api_key"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

7. Deploy.

Do not place the actual API key in GitHub.

## Note about Groq models

Model availability can change. If Groq reports that the configured model is unavailable, replace `GROQ_MODEL` in `.env` or Streamlit Secrets with a model currently available to your Groq account.

## Resume limitation

The parser extracts text from normal PDF/DOCX files. Scanned image-only PDFs may require OCR.
