# Z.E.N.I.T.H. architecture

Z.E.N.I.T.H. = **Z**en **E**ngine for **N**atural **I**nput and **T**ranscription **H**andling.

## Runtime pipeline

1. PortAudio opens the Windows default microphone as mono, signed 16-bit PCM.
2. The program requests 16 kHz. If the default device rejects it, it tries 48 kHz.
3. Audio buffers are passed to Vosk's C API through a dynamically loaded `libvosk.dll`.
4. Vosk partial results are shown while the user is speaking. Vosk finalizes an utterance after endpoint silence.
5. The final `text` string is decoded from Vosk's JSON result and appended to `data/transcripts.txt`.
6. If replay is enabled, microphone capture pauses while eSpeak NG speaks the recognized text, then capture resumes. This reduces feedback from speakers into the microphone.

## Modules

- `audio.c`: PortAudio initialization, microphone stream, PCM capture, shutdown.
- `vosk_dynamic.c`: loads `libvosk.dll` at runtime and resolves the required C API functions.
- `json.c`: small JSON string-property decoder for Vosk's `text` and `partial` properties.
- `tts.c`: launches `espeak-ng.exe` with the transcript as one argument.
- `transcript.c`: timestamped UTF-8 transcript log.
- `main.c`: application lifecycle, CLI options, streaming loop, and replay coordination.

## Design notes and limitations

- This is an offline speech-to-text plus text-to-speech echo/replay application. It speaks the recognized words back using eSpeak NG; it does not reconstruct the speaker's original waveform.
- No cloud service or Python runtime is used.
- A prebuilt Vosk DLL and a separate Vosk acoustic/language model are required. They are intentionally not included in this source ZIP.
- The first version uses the default Windows microphone and a blocking 250 ms audio read. It is designed to be understandable and easy to extend, not to be a production audio engine.
- TTS is invoked as a child process. The mic is stopped during playback to reduce acoustic feedback. Hardware/driver latency can still produce a short gap between utterances.
- The small US English model is a compact starting point; recognition quality depends on microphone, noise, accent, and language model.
