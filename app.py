"""Streamlit UI for Document Extractor MVP"""
import streamlit as st
import tempfile
import json
import csv
import io
from pathlib import Path
from io import BytesIO
from pipeline.extraction_workflow import ExtractionWorkflow
from utils.file_handler import FileHandler


def display_task_results(task_name: str, task_label: str, icon: str, results: dict, validation: dict, tool_calls: list = None):
    """
    Display extraction results and validation for a single task.

    Args:
        task_name: Internal task name (e.g., "author_date")
        task_label: Display label (e.g., "Author & Date")
        icon: Emoji icon for the section
        results: Extraction results for this task
        validation: Validation flags for this task
        tool_calls: List of tool calls made during extraction (for debugging)
    """
    st.subheader(f"{icon} {task_label}")

    # Show extraction results
    st.success("✅ Extraction completed")
    st.json(results)

    # Show validation status
    is_valid = validation.get("confidence") == "high"
    st.checkbox("Validated", value=is_valid, key=f"val_{task_name}", disabled=True)

    # Show tool calls if any
    if tool_calls and len(tool_calls) > 0:
        with st.expander(f"🔍 Tool Usage ({len(tool_calls)} tool{'s' if len(tool_calls) > 1 else ''} called)"):
            for idx, call in enumerate(tool_calls):
                st.write(f"**Tool {idx + 1}: `{call['tool']}`** (Phase: {call.get('phase', 'unknown')})")

                # Show arguments
                st.write("**Arguments:**")
                st.json(call.get("arguments", {}))

                # Show result based on tool type
                if call["tool"] == "validate_date":
                    result = call.get("result", {})
                    if result.get("valid"):
                        st.success(f"✅ Valid date: {result.get('normalized')}")
                    else:
                        st.error(f"❌ Invalid: {result.get('reason')}")

                elif call["tool"] == "expand_keywords_with_web_search":
                    result = call.get("result", {})
                    status = result.get("status", "unknown")

                    if status == "success":
                        st.success(f"✅ {result.get('message')}")
                    elif status == "partial_success":
                        st.warning(f"⚠️ {result.get('message')}")
                    else:
                        st.error(f"❌ {result.get('message')}")

                    # Show original keywords
                    st.write(f"**Original keywords:** {', '.join(result.get('original', []))}")

                    # Show web keywords if any
                    if result.get("web_keywords"):
                        st.write(f"**Web keywords found:** {', '.join(result.get('web_keywords', []))}")

                    # Show sources if any
                    if result.get("sources"):
                        st.write(f"**Sources analyzed ({len(result.get('sources', []))}):**")
                        for source in result.get("sources", []):
                            st.markdown(f"- {source}")

                if idx < len(tool_calls) - 1:
                    st.divider()

    # Show grounding issues if any
    if validation.get("grounding_issues"):
        with st.expander("⚠️ Validation Issues"):
            st.error(f"**Confidence: {validation.get('confidence')}**")
            st.write("**Issues found:**")
            for issue in validation["grounding_issues"]:
                st.warning(issue)
            st.write("**Extracted values that failed grounding:**")
            st.code(json.dumps(results, indent=2))

    st.divider()


# App title
st.title("Document Analyzer")

# File selection - two modes
st.subheader("1. Select Document")
source_mode = st.radio("Document Source:",
                       ["Use Test Image", "Upload File"],
                       horizontal=True)

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
            # Save uploaded file to temporary location for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                uploaded_file.seek(0)
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            try:
                # Run extraction workflow
                workflow = ExtractionWorkflow()

                with st.spinner("Processing document..."):
                    workflow_results = workflow.process_document(tmp_path)

                # Check for critical errors
                if workflow_results["errors"]:
                    st.error("❌ Errors occurred during processing:")
                    for error in workflow_results["errors"]:
                        st.error(error)

                    with st.expander("🐛 Error Details"):
                        st.code("\n".join(workflow_results["errors"]))

                    # Still try to show partial results if available
                    if not workflow_results["ocr_text"]:
                        st.stop()  # Can't continue without OCR text

                # Display OCR text
                with st.expander("Raw OCR Text"):
                    st.text(workflow_results["ocr_text"])

                # Display extracted information
                st.subheader("Extracted Information")

                # Author & Date
                if workflow_results["author_date"]:
                    display_task_results(
                        task_name="author_date",
                        task_label="Author & Date",
                        icon="📄",
                        results=workflow_results["author_date"],
                        validation=workflow_results["validation"]["author_date"],
                        tool_calls=workflow_results.get("tool_calls", {}).get("author_date", [])
                    )

                # Keywords
                if workflow_results["keywords"]:
                    display_task_results(
                        task_name="keywords",
                        task_label="Keywords",
                        icon="🔑",
                        results=workflow_results["keywords"],
                        validation=workflow_results["validation"]["keywords"],
                        tool_calls=workflow_results.get("tool_calls", {}).get("keywords", [])
                    )

                # Document Type
                if workflow_results["document_type"]:
                    display_task_results(
                        task_name="document_type",
                        task_label="Document Type",
                        icon="📋",
                        results=workflow_results["document_type"],
                        validation=workflow_results["validation"]["document_type"],
                        tool_calls=workflow_results.get("tool_calls", {}).get("document_type", [])
                    )

                # Export options
                st.divider()
                st.subheader("Export Results")

                # Prepare results for export
                export_results = {
                    "author_date": workflow_results["author_date"],
                    "keywords": workflow_results["keywords"],
                    "document_type": workflow_results["document_type"]
                }

                # Format text for copy
                handler = FileHandler()
                formatted_text = handler.format_for_copy(export_results)
                st.text_area("Formatted Output", formatted_text, height=150)

                # CSV download button
                csv_buffer = io.StringIO()
                writer = csv.writer(csv_buffer)
                writer.writerow(['Field', 'Value'])
                for key, value in export_results.items():
                    if isinstance(value, dict):
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

            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)

        except Exception as e:
            import traceback
            st.error(f"❌ Critical error: {str(e)}")
            with st.expander("🐛 Critical Error Details"):
                st.code(traceback.format_exc())
