# Z.E.N.I.T.H.

**Z.E.N.I.T.H. — Zen Engine for Natural Input and Transcription Handling**

An offline Windows voice input and spoken-replay module written in C.

- **Speech input:** PortAudio captures microphone audio.
- **Speech recognition:** Vosk converts audio to text locally.
- **Transcript:** recognized utterances are appended to `data/transcripts.txt`.
- **Spoken replay:** eSpeak NG reads the recognized text back.
- **No Python, cloud API, or internet connection is needed at runtime.**

