#ifndef ZENITH_CONFIG_H
#define ZENITH_CONFIG_H

/* Z.E.N.I.T.H. build-time defaults. Paths are relative to project root. */
#define ZENITH_DEFAULT_MODEL_PATH "models/vosk-model-small-en-us-0.15"
#define ZENITH_DEFAULT_VOSK_DLL  "bin/libvosk.dll"
#define ZENITH_DEFAULT_VOICE     "en-us"
#define ZENITH_DEFAULT_RATE      165
#define ZENITH_DEFAULT_SAMPLE_RATE 16000
#define ZENITH_FALLBACK_SAMPLE_RATE 48000
#define ZENITH_FRAMES_PER_BUFFER 4000
#define ZENITH_TEXT_CAPACITY     8192
#define ZENITH_PARTIAL_CAPACITY  4096
#define ZENITH_TRANSCRIPT_PATH   "data/transcripts.txt"

#endif
