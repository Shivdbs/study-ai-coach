import whisper
import librosa


def evaluate_speech(audio_path):
    print(f"Loading audio file: {audio_path}...")

    # 1. NLP Phase: Transcribe with Whisper
    print("Transcribing speech...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    spoken_text = result["text"].strip()

    # 2. DSP Phase: Analyze waveform with Librosa
    print("Analyzing speech flow...")

    # FIX: Use Whisper's internal audio loader instead of Librosa's.
    # Librosa crashes on .m4a files, but Whisper safely converts ANY format
    # into a clean mathematical array using FFmpeg.
    y = whisper.load_audio(audio_path)

    # Whisper always standardizes the sample rate to exactly 16,000 Hz
    sr = 16000

    # Split into non-silent intervals. top_db=20 is the sensitivity threshold.
    intervals = librosa.effects.split(y, top_db=20)

    total_time = librosa.get_duration(y=y, sr=sr)

    # Calculate active speaking time by summing up all the non-silent chunks
    active_time = sum([(end - start) / sr for start, end in intervals])
    pause_time = total_time - active_time

    # 3. Calculate Metrics
    word_count = len(spoken_text.split())
    wpm = (word_count / active_time) * 60 if active_time > 0 else 0
    pause_ratio = pause_time / total_time if total_time > 0 else 0

    # 4. Print Results
    print("\n" + "=" * 40)
    print("🎙️ SPEECH EVALUATION REPORT")
    print("=" * 40)
    print(f"Text Heard : {spoken_text}")
    print(f"Total Time : {total_time:.2f} seconds")
    print(f"Active Time: {active_time:.2f} seconds")
    print(f"Pause Time : {pause_time:.2f} seconds")
    print("-" * 40)
    print(f"Speech Flow: {wpm:.0f} Words Per Minute")
    print(f"Pause Ratio: {pause_ratio:.0%} of total time was silence")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    evaluate_speech("Recording.m4a")