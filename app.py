import streamlit as st
import whisper
import librosa
import tempfile
import os
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI

# Initialize OpenAI Client (You will need to paste your API key here later)
# os.environ["OPENAI_API_KEY"] = "sk-your-api-key-here"
try:
    client = OpenAI()
except:
    client = None

# --- Page Setup ---
st.set_page_config(page_title="Study AI", page_icon="🎙️", layout="centered")

# Create the navigation tabs
tab_speech, tab_notes = st.tabs(["🎙️ Speech Evaluator", "📝 Video Note Maker"])

# ==========================================
# TAB 1: The Speech Engine
# ==========================================
with tab_speech:
    st.header("English Shadowing Coach")
    st.write("Read the phrase below out loud to get feedback on your flow.")

    target_phrase = "She sells seashells by the seashore."
    st.info(f"**Target:** {target_phrase}")

    # Streamlit's native microphone widget
    audio_data = st.audio_input("Record your voice")

    if audio_data is not None:
        with st.spinner("Analyzing your speech flow..."):

            # Save the browser recording to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_data.read())
                tmp_filename = tmp_file.name

            try:
                # 1. Whisper Transcription
                model = whisper.load_model("base")
                result = model.transcribe(tmp_filename)
                spoken_text = result["text"].strip()

                # 2. Librosa Analysis (using Whisper's audio loader to prevent format crashes)
                y = whisper.load_audio(tmp_filename)
                sr = 16000
                intervals = librosa.effects.split(y, top_db=20)

                total_time = librosa.get_duration(y=y, sr=sr)
                active_time = sum([(end - start) / sr for start, end in intervals])
                pause_time = total_time - active_time

                # Calculate Metrics
                word_count = len(spoken_text.split())
                wpm = (word_count / active_time) * 60 if active_time > 0 else 0
                pause_ratio = pause_time / total_time if total_time > 0 else 0

                # 3. Display Results
                st.divider()
                st.write(f"**What we heard:** {spoken_text}")

                col1, col2 = st.columns(2)
                col1.metric("Speech Flow", f"{wpm:.0f} WPM")
                col2.metric("Pause Ratio", f"{pause_ratio:.0%}")

                if pause_ratio > 0.30:
                    st.warning("Feedback: Try to smooth out the gaps between your words.")
                else:
                    st.success("Feedback: Excellent speech flow!")

            finally:
                # Clean up the file
                if os.path.exists(tmp_filename):
                    os.remove(tmp_filename)

# ==========================================
# TAB 2: The Video Summarizer
# ==========================================
with tab_notes:
    st.header("YouTube to Study Notes")
    st.write("Paste a YouTube link below to generate structured study notes instantly.")

    url_input = st.text_input("YouTube URL")

    if st.button("Generate Notes"):
        if not client:
            st.error("OpenAI API Key is missing. Set your environment variable first!")
        elif url_input:
            with st.spinner("Extracting transcript and generating notes..."):
                try:
                    # Extract Video ID
                    video_id = url_input.split("v=")[1].split("&")[0] if "v=" in url_input else \
                    url_input.split("youtu.be/")[1]

                    # Fetch Transcript
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                    full_transcript = " ".join([item['text'] for item in transcript_list])

                    # Send to OpenAI
                    prompt = f"Format this video transcript into structured study notes with a summary, key takeaways, and vocabulary:\n\n{full_transcript[:15000]}"

                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You create structured, easy-to-read study notes."},
                            {"role": "user", "content": prompt}
                        ]
                    )

                    # Display Notes
                    st.divider()
                    st.markdown(response.choices[0].message.content)

                except Exception as e:
                    st.error(f"Error: {e}")