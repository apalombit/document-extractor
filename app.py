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
from chat.context import DocumentContext
from chat.chat_handler import ChatHandler


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
                    elif status == "blocked":
                        st.error(f"🛡️ {result.get('message')}")
                    else:
                        st.error(f"❌ {result.get('message')}")

                    # Show original keywords
                    st.write(f"**Original keywords:** {', '.join(result.get('original', []))}")

                    # Show PII-filtered keywords if any
                    if result.get("pii_filtered"):
                        st.warning(f"🛡️ **PII filtered:** {', '.join(result.get('pii_filtered', []))}")
                        with st.expander("PII Detection Details"):
                            for detail in result.get("pii_details", []):
                                st.write(f"- `{detail.get('keyword')}`: {detail.get('entity_type')} detected (confidence: {detail.get('confidence', 0):.2f})")

                    # Show web keywords if any
                    if result.get("web_keywords"):
                        st.write(f"**Web keywords found:** {', '.join(result.get('web_keywords', []))}")

                    # Show sources if any
                    if result.get("sources"):
                        st.write(f"**Sources analyzed ({len(result.get('sources', []))}):**")
                        for source in result.get("sources", []):
                            st.markdown(f"- {source}")

                    # Show attempted URLs for debugging
                    if result.get("attempted_urls"):
                        with st.expander(f"🔗 URLs attempted ({len(result.get('attempted_urls', []))})"):
                            for url in result.get("attempted_urls", []):
                                st.code(url)

                    # Show fetch errors for debugging
                    if result.get("fetch_errors"):
                        with st.expander(f"⚠️ Fetch errors ({len(result.get('fetch_errors', []))})"):
                            for err in result.get("fetch_errors", []):
                                st.warning(f"{err.get('url')}: {err.get('error')}")

                    # Show debug info if available
                    if result.get("debug_info"):
                        with st.expander("🐛 Debug Info"):
                            debug = result["debug_info"]
                            st.write(f"**Query used:** `{debug.get('query_used', 'N/A')}`")
                            if "raw_results_count" in debug:
                                st.write(f"**Raw results from search:** {debug.get('raw_results_count')}")
                                st.write(f"**After filtering blocked domains:** {debug.get('filtered_count')}")
                            if "pages_with_keywords" in debug:
                                st.write(f"**Pages that yielded keywords:** {debug.get('pages_with_keywords')}")
                            # Show blocked URLs if any
                            if debug.get("blocked_urls"):
                                st.write(f"**URLs blocked:** {len(debug.get('blocked_urls', []))}")
                                for blocked in debug.get("blocked_urls", []):
                                    st.warning(f"`{blocked.get('url', 'N/A')}` - blocked by: `{blocked.get('matched_domain', 'N/A')}`")
                            if "reason" in debug:
                                st.info(debug.get("reason"))
                            if "exception_type" in debug:
                                st.error(f"**Exception:** {debug.get('exception_type')}: {debug.get('exception_message')}")
                            if "traceback" in debug:
                                st.code(debug.get("traceback"))

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


def display_task_error(task_name: str, task_label: str, icon: str, validation: dict):
    """Display error state for a failed task."""
    st.subheader(f"{icon} {task_label}")
    st.error("❌ Extraction failed")

    if validation.get("grounding_issues"):
        st.write("**Error details:**")
        for issue in validation["grounding_issues"]:
            st.code(issue)
    else:
        st.warning("No error details available")

    # Show raw LLM response if available (helps debug JSON parse errors)
    if validation.get("raw_response"):
        with st.expander("🔍 Raw LLM Response (failed to parse)"):
            st.code(validation["raw_response"])

    # Show full traceback if available
    if validation.get("traceback"):
        with st.expander("🐛 Full Exception Traceback"):
            st.code(validation["traceback"])

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
    # Thumbnail with option to expand
    col1, col2 = st.columns([1, 3])
    with col1:
        expand_image = st.checkbox("Full size", value=False)

    if expand_image:
        st.image(uploaded_file, caption="Document Preview (Full Size)")
    else:
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
                with st.expander("Raw OCR Text", expanded=True):
                    st.text(workflow_results["ocr_text"])

                # Display extracted information
                st.subheader("Extracted Information")

                # Author & Date - always show
                if workflow_results["author_date"]:
                    display_task_results(
                        task_name="author_date",
                        task_label="Author & Date",
                        icon="📄",
                        results=workflow_results["author_date"],
                        validation=workflow_results["validation"]["author_date"],
                        tool_calls=workflow_results.get("tool_calls", {}).get("author_date", [])
                    )
                else:
                    display_task_error(
                        task_name="author_date",
                        task_label="Author & Date",
                        icon="📄",
                        validation=workflow_results["validation"]["author_date"]
                    )

                # Keywords - always show
                if workflow_results["keywords"]:
                    display_task_results(
                        task_name="keywords",
                        task_label="Keywords",
                        icon="🔑",
                        results=workflow_results["keywords"],
                        validation=workflow_results["validation"]["keywords"],
                        tool_calls=workflow_results.get("tool_calls", {}).get("keywords", [])
                    )
                else:
                    display_task_error(
                        task_name="keywords",
                        task_label="Keywords",
                        icon="🔑",
                        validation=workflow_results["validation"]["keywords"]
                    )

                # Document Type - always show
                if workflow_results["document_type"]:
                    display_task_results(
                        task_name="document_type",
                        task_label="Document Type",
                        icon="📋",
                        results=workflow_results["document_type"],
                        validation=workflow_results["validation"]["document_type"],
                        tool_calls=workflow_results.get("tool_calls", {}).get("document_type", [])
                    )
                else:
                    display_task_error(
                        task_name="document_type",
                        task_label="Document Type",
                        icon="📋",
                        validation=workflow_results["validation"]["document_type"]
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

                # Store workflow results in session state for chat
                st.session_state.workflow_results = workflow_results

                # Initialize chat handler
                context = DocumentContext.from_workflow_results(workflow_results)
                st.session_state.chat_handler = ChatHandler(context)
                st.session_state.chat_messages = []

            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)

        except Exception as e:
            import traceback
            st.error(f"❌ Critical error: {str(e)}")
            with st.expander("🐛 Critical Error Details"):
                st.code(traceback.format_exc())

# Chat section - available after document is analyzed
if "chat_handler" in st.session_state and st.session_state.chat_handler is not None:
    st.divider()
    st.subheader("Chat with Document")

    # Two-column layout: OCR reference on left, chat on right
    ocr_col, chat_col = st.columns([1, 1])

    with ocr_col:
        st.caption("OCR Text Reference")
        # Show OCR text in a scrollable container
        if "workflow_results" in st.session_state:
            ocr_text = st.session_state.workflow_results.get("ocr_text", "")
            st.text_area("", ocr_text, height=400, disabled=True, label_visibility="collapsed")

    with chat_col:
        st.caption("Ask questions about the analyzed document")

        # Initialize chat messages if not present
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        # Chat container with fixed height for scrolling
        chat_container = st.container(height=350)

        with chat_container:
            # Display chat history
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask about this document..."):
            # Add user message to chat history
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            # Get response from chat handler
            with st.spinner("Thinking..."):
                response = st.session_state.chat_handler.chat(prompt)

            # Add assistant response to chat history
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()

        # Clear chat button
        if st.session_state.chat_messages:
            if st.button("Clear Chat", type="secondary"):
                st.session_state.chat_messages = []
                st.session_state.chat_handler.clear_history()
                st.rerun()

    # Debug section (full width below)
    with st.expander("Debug: Chat Context"):
        debug_info = st.session_state.chat_handler.get_debug_info()
        st.write("**Chat State:**")
        st.json(debug_info)

        st.write("**Messages sent to LLM:**")
        provider_messages = st.session_state.chat_handler.get_provider_messages()
        for i, msg in enumerate(provider_messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate long content for display
            preview = content[:500] + "..." if len(content) > 500 else content
            with st.container():
                st.markdown(f"**[{i+1}] {role.upper()}**")
                st.code(preview, language=None)
