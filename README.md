# 🎙️ Study AI: Speech Analytics & Video Summarizer

**Live Demo:** [Click here to view the live application](https://your-app-url.streamlit.app)

## Overview
Study AI is an end-to-end full-stack web application designed to evaluate spoken English fluency and generate instant study notes from video lectures. It combines Natural Language Processing (NLP) with Digital Signal Processing (DSP) to extract both textual meaning and physical vocal pacing from raw audio data.

## Features
* **English Shadowing Coach:** Captures real-time browser audio and calculates speech metrics (Words Per Minute and Pause/Silence Ratios).
* **Automated Video Summarizer:** Bypasses manual transcription by extracting YouTube subtitles and utilizing Generative AI to structure hour-long lectures into concise study notes.
* **Serverless Deployment:** Built with a lightweight architecture optimized for cloud deployment.

## Tech Stack
* **Frontend & Backend Framework:** `Streamlit`, `Python`
* **Natural Language Processing:** `OpenAI Whisper` (Local STT), `OpenAI GPT-3.5` (LLM)
* **Digital Signal Processing (DSP):** `Librosa`, `NumPy`
* **Data Extraction:** `youtube-transcript-api`

## System Architecture & Logic
1. **Audio Processing:** Instead of relying solely on text output, the app converts `.wav`/`.m4a` files into mathematical arrays. It isolates non-silent intervals using a `20db` threshold, allowing the system to accurately measure hesitations and calculate physical delivery metrics.
2. **Format Handling:** Utilizes Whisper's built-in FFmpeg integration to safely load varied browser audio formats into memory, bypassing standard `soundfile` format crashes.
3. **Prompt Engineering:** The LLM integration utilizes strict system prompting to ensure video notes are consistently returned in a standardized, academic format (Summary, Key Takeaways, Vocabulary).
