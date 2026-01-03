# Manual Testing Guide for Phase 4 - Streamlit UI

## Prerequisites

1. **Start Ollama service** (required for LLM extraction):
   ```bash
   ollama serve
   ```

2. **Activate virtual environment and run app**:
   ```bash
   source venv/bin/activate
   streamlit run app.py
   ```

3. The app should open in your browser at `http://localhost:8501`

## New Debugging Features

The UI now includes enhanced troubleshooting tools:

### 1. Test Image Selector
- Select "Use Test Image" radio button
- Choose from available test images in dropdown (from `tests/fixtures/sample_documents/`)
- No need to drag & drop files manually

### 2. Debug: Prompts & Response
Each extraction task includes a "🔍 Debug: Prompts & Response" expandable section showing:
- **System Prompt**: The full system prompt sent to LLM (including OCR text)
- **User Prompt**: The task-specific prompt with examples and schema
- **Raw LLM Response**: The exact JSON response from Ollama before parsing

### 3. Enhanced Validation Issues
When grounding check fails, the "⚠️ Validation Issues" section now shows:
- **Confidence level**: High or Low
- **Specific issues**: Which extracted values were not found in source text
- **Failed values**: The complete extracted data structure that failed validation

## Test Cases

### Test Case 1: Happy Path - Valid Medical Document ✅

**Steps:**
1. Select "Use Test Image" radio button
2. Choose "medical_report.png" from dropdown
3. Verify image preview displays correctly
4. Click "Analyze Document"
5. Wait for OCR processing
6. Verify raw OCR text appears in expandable section
7. Wait for 3 LLM extraction tasks (author_date, keywords, document_type)
8. Verify JSON results display for each task
9. Check validation checkboxes show confidence level
10. Expand "🔍 Debug: Prompts & Response" for each task to inspect:
    - System prompt includes OCR text
    - User prompt has task-specific instructions
    - Raw LLM response is valid JSON
11. Check if any "⚠️ Validation Issues" expanders appear
12. Verify formatted output appears in text area
13. Click "Download CSV" button to test CSV export
14. Verify no errors in "⚠️ Errors" section at bottom

**Expected Results:**
- All extraction tasks complete successfully
- High confidence validation (checkboxes checked)
- No validation issues (or minimal issues)
- Debug sections show complete prompt chain
- Clean export functionality

---

### Test Case 2: Invalid File Format ⚠️

**Steps:**
1. Try to upload a .jpg or .txt file

**Expected Results:**
- File uploader should reject non-PNG files (only .png accepted)

---

### Test Case 3: Empty/Blank Document 🔍

**Steps:**
1. Create a blank white PNG image:
   ```python
   from PIL import Image
   img = Image.new('RGB', (800, 600), color='white')
   img.save('blank.png')
   ```
2. Upload the blank image
3. Click "Analyze Document"

**Expected Results:**
- OCR completes (returns empty or minimal text)
- LLM tasks handle empty text gracefully
- Validation shows low confidence or null results
- No crashes

---

### Test Case 4: Large File (>10MB) 📦

**Steps:**
1. Create a large PNG (>10MB):
   ```python
   from PIL import Image
   img = Image.new('RGB', (10000, 10000), color='white')
   img.save('large.png')
   ```
2. Upload the large file

**Expected Results:**
- Error message appears: "File too large: X.XMB"
- Error displayed in "⚠️ Errors" section
- App remains functional

---

### Test Case 5: Ollama Connection Failure ❌

**Steps:**
1. Stop Ollama service: `killall ollama` or close Ollama app
2. Upload valid medical document
3. Click "Analyze Document"

**Expected Results:**
- OCR completes successfully
- LLM extraction tasks fail with connection errors
- Errors appear in "⚠️ Errors" section
- App remains functional (doesn't crash)
- Can still see OCR text

---

### Test Case 6: Grounding Validation Display 🔎

**Steps:**
1. Upload medical document
2. After extraction, look for validation checkboxes
3. Check if any "⚠️ Validation Issues" expanders appear
4. If they appear, expand them to see grounding issues

**Expected Results:**
- Validation checkboxes reflect confidence level (checked = high, unchecked = low)
- If LLM hallucinates content not in OCR text, validation issues appear
- Issues clearly state which values were not found in source text

---

## Known Limitations (MVP)

1. **Single document processing only** - No batch upload
2. **Manual testing only** - No automated UI tests (Streamlit limitation)
3. **No retry mechanism** - If extraction fails, must re-upload
4. **Local only** - Not deployed to cloud
5. **PNG only** - No PDF, JPG, or other formats
6. **File size limit** - 10MB maximum

---

## Future Enhancements (Post-MVP)

1. **Automated UI Testing** - Consider Selenium or Playwright
2. **Batch Processing** - Multiple documents at once
3. **PDF Support** - Extract from PDFs
4. **Cloud Deployment** - Deploy to Streamlit Cloud
5. **User Authentication** - Multi-user support
6. **Data Persistence** - Save extraction history
7. **Manual Validation Interface** - Allow users to correct extractions
8. **HTML Generation** - Convert documents to HTML (stretch goal)
