import streamlit as st
import pdfplumber
from transformers import pipeline
import textwrap

# Load summarization pipeline
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# App title
st.title("📘 Policy Brief Generator")
st.subheader("Turn complex education policy PDFs into clear, parent-friendly summaries.")

# Upload PDF
uploaded_file = st.file_uploader("Upload an education policy PDF", type="pdf")

# Summary length option
summary_length = st.selectbox("Choose summary length:", ["Short (2-3 paragraphs)", "Medium (5 paragraphs)", "Detailed (8+ paragraphs)"])

length_settings = {
    "Short (2-3 paragraphs)": {"max_length": 150, "min_length": 60},
    "Medium (5 paragraphs)": {"max_length": 250, "min_length": 100},
    "Detailed (8+ paragraphs)": {"max_length": 350, "min_length": 150}
}

if uploaded_file is not None:
    st.info("Extracting text from your PDF...")

    # Extract text
    with pdfplumber.open(uploaded_file) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    if full_text:
        st.success("Text extracted. Generating summary...")

        # Prepend a helpful instruction to improve summary quality
        prompt_prefix = (
            "Summarize this clearly so a parent can understand their rights and responsibilities in plain language:\n\n"
        )

        # Chunk and summarize
        chunks = textwrap.wrap(prompt_prefix + full_text, width=1000)
        summary_parts = []

        for chunk in chunks[:3]:  # summarize only first 3 chunks
            result = summarizer(
                chunk,
                max_length=length_settings[summary_length]["max_length"],
                min_length=length_settings[summary_length]["min_length"],
                do_sample=False
            )
            summary_parts.append(result[0]['summary_text'])

        final_summary = "\n\n".join(summary_parts)

        st.markdown("### 📝 Policy Brief")
        st.write(final_summary)

        import io

        st.download_button(
            label="📥 Download Summary as TXT",
            data=io.StringIO(final_summary).getvalue(),
            file_name="policy_summary.txt",
            mime="text/plain"
        )

        # from gtts import gTTS
        # import os

        # # Convert summary text to speech
        # tts = gTTS(text=final_summary)
        # tts.save("summary.mp3")

        # # Add audio player
        # audio_file = open("summary.mp3", "rb")
        # audio_bytes = audio_file.read()
        # st.audio(audio_bytes, format="audio/mp3")

    else:
        st.warning("No readable text found in the PDF.")
