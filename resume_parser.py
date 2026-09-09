from io import BytesIO

from pypdf import PdfReader
from docx import Document


class ResumeParserError(Exception):
    """Raised when a resume cannot be parsed."""


class ResumeParser:
    """Responsible only for extracting text from resume files."""

    @staticmethod
    def extract_text(file_bytes: bytes, filename: str) -> str:
        if not file_bytes:
            raise ResumeParserError("The uploaded file is empty.")

        extension = filename.lower().rsplit(".", 1)[-1]

        try:
            if extension == "pdf":
                return ResumeParser._extract_pdf(file_bytes)

            if extension == "docx":
                return ResumeParser._extract_docx(file_bytes)

            raise ResumeParserError(
                "Unsupported file type. Please upload PDF or DOCX."
            )

        except ResumeParserError:
            raise
        except Exception as exc:
            raise ResumeParserError(
                f"Could not read the resume: {exc}"
            ) from exc

    @staticmethod
    def _extract_pdf(file_bytes: bytes) -> str:
        reader = PdfReader(BytesIO(file_bytes))
        pages = []

        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())

        text = "\n\n".join(pages).strip()

        if not text:
            raise ResumeParserError(
                "No readable text was found in the PDF."
            )

        return text

    @staticmethod
    def _extract_docx(file_bytes: bytes) -> str:
        document = Document(BytesIO(file_bytes))
        parts = []

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if text:
                parts.append(text)

        for table in document.tables:
            for row in table.rows:
                cells = [
                    cell.text.strip()
                    for cell in row.cells
                    if cell.text.strip()
                ]
                if cells:
                    parts.append(" | ".join(cells))

        text = "\n".join(parts).strip()

        if not text:
            raise ResumeParserError(
                "No readable text was found in the DOCX file."
            )

        return text
