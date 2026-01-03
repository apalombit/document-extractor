"""Streamlit UI for Document Extractor MVP"""
import streamlit as st
import tempfile
from pathlib import Path
from utils.file_handler import FileHandler
from ocr.engine import SelectedOCR
from llm.extractor import LLMExtractor

st.title("Document Analyzer")

# File selection - two modes
st.subheader("1. Select Document")
source_mode = st.radio("Document Source:", ["Upload File", "Use Test Image"], horizontal=True)

uploaded_file = None

if source_mode == "Upload File":
    uploaded_file = st.file_uploader("Select document", type=['png'])
else:
    # List test images from fixtures
    test_images_dir = Path("tests/fixtures/sample_documents")
    if test_images_dir.exists():
        test_images = list(test_images_dir.glob("*.png"))
        if test_images:
            test_image_names = [img.name for img in test_images]
            selected_image = st.selectbox("Choose test image:", test_image_names)
            if selected_image:
                test_image_path = test_images_dir / selected_image
                # Create a file-like object from the test image
                with open(test_image_path, 'rb') as f:
                    from io import BytesIO
                    uploaded_file = BytesIO(f.read())
                    uploaded_file.name = selected_image
        else:
            st.warning("No test images found in tests/fixtures/sample_documents/")
    else:
        st.error("Test images directory not found: tests/fixtures/sample_documents/")

if uploaded_file:
    st.image(uploaded_file, caption="Document Preview", width=300)

    if st.button("Analyze Document"):
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
            st.subheader("📄 Author & Date")
            system_prompt = None
            user_prompt = None
            raw_response = None
            error_info = None

            try:
                with st.spinner("Extracting author and date..."):
                    # Get prompts for debugging
                    system_prompt = extractor.system_prompt.format(ocr_text=ocr_text)
                    user_prompt = extractor.task_prompts["author_date"]

                    # Call LLM
                    raw_response = extractor.provider.generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=extractor.temperature,
                        max_tokens=extractor.max_tokens
                    )

                    # Parse and validate
                    import json
                    results["author_date"] = json.loads(raw_response)
                    validation_flags["author_date"] = extractor._validate_grounding(
                        results["author_date"], ocr_text
                    )

                st.success("✅ Extraction completed")
                st.json(results["author_date"])

                # Show validation status
                is_valid = validation_flags["author_date"]["confidence"] == "high"
                st.checkbox("Validated", value=is_valid, key="val_author", disabled=True)

                # Show grounding issues if any
                if validation_flags["author_date"]["grounding_issues"]:
                    with st.expander("⚠️ Validation Issues"):
                        st.error(f"**Confidence: {validation_flags['author_date']['confidence']}**")
                        st.write("**Issues found:**")
                        for issue in validation_flags["author_date"]["grounding_issues"]:
                            st.warning(issue)
                        st.write("**Extracted values that failed grounding:**")
                        st.code(json.dumps(results["author_date"], indent=2))

            except Exception as e:
                import traceback
                error_info = {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                st.error(f"❌ Extraction failed: {str(e)}")

                with st.expander("🐛 Error Details"):
                    st.code(error_info["traceback"])
                    if raw_response is not None:
                        st.write("**Raw LLM Response (before parsing):**")
                        st.code(raw_response)

            # Debug: Show prompts and response (always show, even on error)
            with st.expander("🔍 Debug: Prompts & Response"):
                if system_prompt:
                    st.write("**System Prompt:**")
                    st.text_area("System", system_prompt, height=150, key="sys_author")
                if user_prompt:
                    st.write("**User Prompt:**")
                    st.text_area("User", user_prompt, height=100, key="user_author")
                if raw_response is not None:
                    st.write("**Raw LLM Response:**")
                    st.code(raw_response, language="json")
                else:
                    st.warning("No response received from LLM")

            st.divider()
            
            # Keywords
            st.subheader("🔑 Keywords")
            system_prompt = None
            user_prompt = None
            raw_response = None
            error_info = None

            try:
                with st.spinner("Extracting keywords..."):
                    # Get prompts for debugging
                    system_prompt = extractor.system_prompt.format(ocr_text=ocr_text)
                    user_prompt = extractor.task_prompts["keywords"]

                    # Call LLM
                    raw_response = extractor.provider.generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=extractor.temperature,
                        max_tokens=extractor.max_tokens
                    )

                    # Parse and validate
                    results["keywords"] = json.loads(raw_response)
                    validation_flags["keywords"] = extractor._validate_grounding(
                        results["keywords"], ocr_text
                    )

                st.success("✅ Extraction completed")
                st.json(results["keywords"])

                # Show validation status
                is_valid = validation_flags["keywords"]["confidence"] == "high"
                st.checkbox("Validated", value=is_valid, key="val_keywords", disabled=True)

                # Show grounding issues if any
                if validation_flags["keywords"]["grounding_issues"]:
                    with st.expander("⚠️ Validation Issues"):
                        st.error(f"**Confidence: {validation_flags['keywords']['confidence']}**")
                        st.write("**Issues found:**")
                        for issue in validation_flags["keywords"]["grounding_issues"]:
                            st.warning(issue)
                        st.write("**Extracted values that failed grounding:**")
                        st.code(json.dumps(results["keywords"], indent=2))

            except Exception as e:
                import traceback
                error_info = {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                st.error(f"❌ Extraction failed: {str(e)}")

                with st.expander("🐛 Error Details"):
                    st.code(error_info["traceback"])
                    if raw_response is not None:
                        st.write("**Raw LLM Response (before parsing):**")
                        st.code(raw_response)

            # Debug: Show prompts and response (always show, even on error)
            with st.expander("🔍 Debug: Prompts & Response"):
                if system_prompt:
                    st.write("**System Prompt:**")
                    st.text_area("System", system_prompt, height=150, key="sys_keywords")
                if user_prompt:
                    st.write("**User Prompt:**")
                    st.text_area("User", user_prompt, height=100, key="user_keywords")
                if raw_response is not None:
                    st.write("**Raw LLM Response:**")
                    st.code(raw_response, language="json")
                else:
                    st.warning("No response received from LLM")

            st.divider()
            
            # Document Type
            st.subheader("📋 Document Type")
            system_prompt = None
            user_prompt = None
            raw_response = None
            error_info = None

            try:
                with st.spinner("Classifying document..."):
                    # Get prompts for debugging
                    system_prompt = extractor.system_prompt.format(ocr_text=ocr_text)
                    user_prompt = extractor.task_prompts["document_type"]

                    # Call LLM
                    raw_response = extractor.provider.generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=extractor.temperature,
                        max_tokens=extractor.max_tokens
                    )

                    # Parse and validate
                    results["document_type"] = json.loads(raw_response)
                    validation_flags["document_type"] = extractor._validate_grounding(
                        results["document_type"], ocr_text
                    )

                st.success("✅ Extraction completed")
                st.json(results["document_type"])

                # Show validation status
                is_valid = validation_flags["document_type"]["confidence"] == "high"
                st.checkbox("Validated", value=is_valid, key="val_doctype", disabled=True)

                # Show grounding issues if any
                if validation_flags["document_type"]["grounding_issues"]:
                    with st.expander("⚠️ Validation Issues"):
                        st.error(f"**Confidence: {validation_flags['document_type']['confidence']}**")
                        st.write("**Issues found:**")
                        for issue in validation_flags["document_type"]["grounding_issues"]:
                            st.warning(issue)
                        st.write("**Extracted values that failed grounding:**")
                        st.code(json.dumps(results["document_type"], indent=2))

            except Exception as e:
                import traceback
                error_info = {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                st.error(f"❌ Extraction failed: {str(e)}")

                with st.expander("🐛 Error Details"):
                    st.code(error_info["traceback"])
                    if raw_response is not None:
                        st.write("**Raw LLM Response (before parsing):**")
                        st.code(raw_response)

            # Debug: Show prompts and response (always show, even on error)
            with st.expander("🔍 Debug: Prompts & Response"):
                if system_prompt:
                    st.write("**System Prompt:**")
                    st.text_area("System", system_prompt, height=150, key="sys_doctype")
                if user_prompt:
                    st.write("**User Prompt:**")
                    st.text_area("User", user_prompt, height=100, key="user_doctype")
                if raw_response is not None:
                    st.write("**Raw LLM Response:**")
                    st.code(raw_response, language="json")
                else:
                    st.warning("No response received from LLM")

            st.divider()
            
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
            import traceback
            st.error(f"❌ Critical error: {str(e)}")
            with st.expander("🐛 Critical Error Details"):
                st.code(traceback.format_exc())
