"""Streamlit UI for Document Extractor MVP"""
import streamlit as st
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
            # OCR Processing
            with st.spinner("Extracting text..."):
                handler = FileHandler()
                image = handler.load_image(uploaded_file)
                ocr = SelectedOCR()
                ocr_text = ocr.extract_text(image)
            
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
                st.checkbox("Validated", 
                           value=validation_flags["author_date"]["confidence"]=="high", 
                           key="val_author", disabled=True)
            except Exception as e:
                st.session_state.errors.append(f"Author/Date: {str(e)}")
            
            # Keywords
            try:
                with st.spinner("Extracting keywords..."):
                    results["keywords"], validation_flags["keywords"] = \
                        extractor.extract_field(ocr_text, "keywords")
                st.success("Keywords")
                st.json(results["keywords"])
                st.checkbox("Validated", 
                           value=validation_flags["keywords"]["confidence"]=="high",
                           key="val_keywords", disabled=True)
            except Exception as e:
                st.session_state.errors.append(f"Keywords: {str(e)}")
            
            # Document Type
            try:
                with st.spinner("Classifying document..."):
                    results["document_type"], validation_flags["document_type"] = \
                        extractor.extract_field(ocr_text, "document_type")
                st.success("Document Type")
                st.json(results["document_type"])
                st.checkbox("Validated", 
                           value=validation_flags["document_type"]["confidence"]=="high",
                           key="val_doctype", disabled=True)
            except Exception as e:
                st.session_state.errors.append(f"Document Type: {str(e)}")
            
            # Export options
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Copy Text"):
                    text = handler.format_for_copy(results)
                    st.code(text)
            
            with col2:
                if st.button("Download CSV"):
                    handler.export_to_csv(results, "output.csv")
                    st.success("Saved to output.csv")
                    
        except Exception as e:
            st.session_state.errors.append(f"Critical error: {str(e)}")

# Error tab
if st.session_state.errors:
    with st.expander("⚠️ Errors", expanded=True):
        for error in st.session_state.errors:
            st.error(error)
