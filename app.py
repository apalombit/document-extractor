"""Streamlit UI for Document Extractor MVP"""
import streamlit as st
import tempfile
from pathlib import Path
from utils.file_handler import FileHandler
from ocr.engine import SelectedOCR
from llm.extractor import LLMExtractor

st.title("Document Analyzer")

# Error tracking
if 'errors' not in st.session_state:
    st.session_state.errors = []

# File selection
uploaded_file = st.file_uploader("Select document", type=['png'])

if uploaded_file:
    st.image(uploaded_file, caption="Document Preview", width=300)

    if st.button("Analyze Document"):
        st.session_state.errors = []  # Reset errors

        try:
            # Validate file first
            handler = FileHandler()
            image = handler.load_image(uploaded_file)

            # OCR Processing - save to temp file for OCR
            with st.spinner("Extracting text..."):
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    uploaded_file.seek(0)
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                try:
                    ocr = SelectedOCR()
                    ocr_text = ocr.extract_text(tmp_path)
                finally:
                    # Clean up temp file
                    Path(tmp_path).unlink(missing_ok=True)
            
            with st.expander("Raw OCR Text"):
                st.text(ocr_text)
            
            # LLM extraction
            st.subheader("Extracted Information")
            extractor = LLMExtractor()
            results = {}
            validation_flags = {}
            
            # Author & Date
            try:
                with st.spinner("Extracting author and date..."):
                    results["author_date"], validation_flags["author_date"] = \
                        extractor.extract_field(ocr_text, "author_date")
                st.success("Author & Date")
                st.json(results["author_date"])

                # Show validation status
                is_valid = validation_flags["author_date"]["confidence"] == "high"
                st.checkbox("Validated", value=is_valid, key="val_author", disabled=True)

                # Show grounding issues if any
                if validation_flags["author_date"]["grounding_issues"]:
                    with st.expander("⚠️ Validation Issues"):
                        for issue in validation_flags["author_date"]["grounding_issues"]:
                            st.warning(issue)
            except Exception as e:
                st.session_state.errors.append(f"Author/Date: {str(e)}")
            
            # Keywords
            try:
                with st.spinner("Extracting keywords..."):
                    results["keywords"], validation_flags["keywords"] = \
                        extractor.extract_field(ocr_text, "keywords")
                st.success("Keywords")
                st.json(results["keywords"])

                # Show validation status
                is_valid = validation_flags["keywords"]["confidence"] == "high"
                st.checkbox("Validated", value=is_valid, key="val_keywords", disabled=True)

                # Show grounding issues if any
                if validation_flags["keywords"]["grounding_issues"]:
                    with st.expander("⚠️ Validation Issues"):
                        for issue in validation_flags["keywords"]["grounding_issues"]:
                            st.warning(issue)
            except Exception as e:
                st.session_state.errors.append(f"Keywords: {str(e)}")
            
            # Document Type
            try:
                with st.spinner("Classifying document..."):
                    results["document_type"], validation_flags["document_type"] = \
                        extractor.extract_field(ocr_text, "document_type")
                st.success("Document Type")
                st.json(results["document_type"])

                # Show validation status
                is_valid = validation_flags["document_type"]["confidence"] == "high"
                st.checkbox("Validated", value=is_valid, key="val_doctype", disabled=True)

                # Show grounding issues if any
                if validation_flags["document_type"]["grounding_issues"]:
                    with st.expander("⚠️ Validation Issues"):
                        for issue in validation_flags["document_type"]["grounding_issues"]:
                            st.warning(issue)
            except Exception as e:
                st.session_state.errors.append(f"Document Type: {str(e)}")
            
            # Export options
            st.divider()
            st.subheader("Export Results")

            # Format text for copy
            formatted_text = handler.format_for_copy(results)
            st.text_area("Formatted Output", formatted_text, height=150)

            # CSV download button
            import csv
            import io
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(['Field', 'Value'])
            for key, value in results.items():
                if isinstance(value, dict):
                    import json
                    writer.writerow([key, json.dumps(value)])
                elif isinstance(value, list):
                    writer.writerow([key, ', '.join(map(str, value))])
                else:
                    writer.writerow([key, value])

            st.download_button(
                label="Download CSV",
                data=csv_buffer.getvalue(),
                file_name="extracted_data.csv",
                mime="text/csv"
            )
                    
        except Exception as e:
            st.session_state.errors.append(f"Critical error: {str(e)}")

# Error tab
if st.session_state.errors:
    with st.expander("⚠️ Errors", expanded=True):
        for error in st.session_state.errors:
            st.error(error)
