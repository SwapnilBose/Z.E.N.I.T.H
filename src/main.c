#define _CRT_SECURE_NO_WARNINGS

#include "zenith_config.h"
#include "zenith_audio.h"
#include "zenith_vosk.h"
#include "zenith_json.h"
#include "zenith_tts.h"
#include "zenith_transcript.h"

#include <windows.h>
#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *program)
{
    printf("\nZ.E.N.I.T.H. - Zen's Engine for Natural Input and Transcription Handling\n\n");
    printf("Usage: %s [options]\n\n", program);
    printf("Options:\n");
    printf("  --model PATH       Vosk model directory\n");
    printf("  --vosk-dll PATH    Vosk DLL path (default: %s)\n", ZENITH_DEFAULT_VOSK_DLL);
    printf("  --voice NAME       eSpeak voice (default: %s)\n", ZENITH_DEFAULT_VOICE);
    printf("  --rate WPM         eSpeak speaking rate, 80-450 (default: %d)\n", ZENITH_DEFAULT_RATE);
    printf("  --no-replay        Transcribe without speaking the text back\n");
    printf("  --help             Show this help\n\n");
    printf("While listening, press Q or ESC to exit. Press other keys only when idle.\n");
}

static int path_exists(const char *path)
{
    DWORD attributes = GetFileAttributesA(path);
    return attributes != INVALID_FILE_ATTRIBUTES;
}

static void print_partial_if_changed(const char *json, char *last_partial,
                                     size_t last_cap)
{
    char partial[ZENITH_PARTIAL_CAPACITY];
    if (!zenith_json_get_string(json, "partial", partial, sizeof(partial)))
        return;
    if (partial[0] == '\0' || strcmp(partial, last_partial) == 0)
        return;

    snprintf(last_partial, last_cap, "%s", partial);
    printf("\r  ... %s", partial);
    fflush(stdout);
}

static void handle_utterance(const char *json, ZenithAudio *audio,
                             const char *voice, int rate, int replay)
{
    char text[ZENITH_TEXT_CAPACITY];
    char audio_error[256];

    if (!zenith_json_get_string(json, "text", text, sizeof(text)))
        return;

    if (text[0] == '\0')
        return;

    printf("\r\nYOU: %s\n", text);
    if (!zenith_transcript_append(text, ZENITH_TRANSCRIPT_PATH)) {
        fprintf(stderr, "[Z.E.N.I.T.H.] Warning: transcript could not be saved.\n");
    }

    if (replay) {
        /* Pause capture while the speakers talk, preventing the TTS output
           from being immediately recognized by the microphone. */
        if (!zenith_audio_stop(audio, audio_error, sizeof(audio_error))) {
            fprintf(stderr, "[Z.E.N.I.T.H.] Could not pause microphone: %s\n", audio_error);
        }

        printf("ZENITH (spoken replay): %s\n", text);
        zenith_speak(text, voice, rate);

        if (!zenith_audio_start(audio, audio_error, sizeof(audio_error))) {
            fprintf(stderr, "[Z.E.N.I.T.H.] Could not restart microphone: %s\n", audio_error);
        }
    }
}

int main(int argc, char **argv)
{
    const char *model_path = ZENITH_DEFAULT_MODEL_PATH;
    const char *vosk_dll_path = ZENITH_DEFAULT_VOSK_DLL;
    const char *voice = ZENITH_DEFAULT_VOICE;
    int rate = ZENITH_DEFAULT_RATE;
    int replay = 1;
    int i;

    ZenithVosk vosk;
    ZenithAudio audio;
    VoskModel *model = NULL;
    VoskRecognizer *recognizer = NULL;
    short samples[ZENITH_FRAMES_PER_BUFFER];
    char error[512];
    char last_partial[ZENITH_PARTIAL_CAPACITY] = "";
    int running = 1;

    memset(&vosk, 0, sizeof(vosk));
    memset(&audio, 0, sizeof(audio));

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--vosk-dll") == 0 && i + 1 < argc) {
            vosk_dll_path = argv[++i];
        } else if (strcmp(argv[i], "--voice") == 0 && i + 1 < argc) {
            voice = argv[++i];
        } else if (strcmp(argv[i], "--rate") == 0 && i + 1 < argc) {
            rate = atoi(argv[++i]);
            if (rate < 80 || rate > 450) {
                fprintf(stderr, "Speaking rate must be between 80 and 450.\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--no-replay") == 0) {
            replay = 0;
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }

    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    printf("============================================================\n");
    printf(" Z.E.N.I.T.H.\n");
    printf(" Zen's Engine for Natural Input and Transcription Handling\n");
    printf("============================================================\n");
    printf("Mode: continuous offline speech recognition\n");
    printf("Replay: %s\n", replay ? "enabled (eSpeak NG)" : "disabled");

    if (!path_exists(model_path)) {
        fprintf(stderr, "\nModel directory not found: %s\n", model_path);
        fprintf(stderr, "Extract your Vosk model into the models/ directory. See README.md.\n");
        return 1;
    }

    if (!zenith_vosk_load(&vosk, vosk_dll_path, error, sizeof(error))) {
        fprintf(stderr, "\nVosk initialization failed: %s\n", error);
        return 1;
    }

    printf("\nLoading Vosk model (this may take a little while)...\n");
    model = zenith_vosk_model_new(&vosk, model_path);
    if (!model) {
        fprintf(stderr, "Could not load Vosk model from: %s\n", model_path);
        zenith_vosk_unload(&vosk);
        return 1;
    }
    printf("Vosk model loaded.\n");

    if (!zenith_audio_open(&audio, error, sizeof(error))) {
        fprintf(stderr, "Microphone initialization failed: %s\n", error);
        vosk.model_free(model);
        zenith_vosk_unload(&vosk);
        return 1;
    }

    recognizer = zenith_vosk_recognizer_new(&vosk, model, (float)audio.sample_rate);
    if (!recognizer) {
        fprintf(stderr, "Could not create Vosk recognizer.\n");
        zenith_audio_close(&audio);
        vosk.model_free(model);
        zenith_vosk_unload(&vosk);
        return 1;
    }

    printf("Microphone sample rate: %d Hz, mono, signed 16-bit PCM\n", audio.sample_rate);

    if (!zenith_audio_start(&audio, error, sizeof(error))) {
        fprintf(stderr, "Could not start microphone: %s\n", error);
        vosk.recognizer_free(recognizer);
        zenith_audio_close(&audio);
        vosk.model_free(model);
        zenith_vosk_unload(&vosk);
        return 1;
    }

    if (replay && !zenith_espeak_available()) {
        fprintf(stderr, "Warning: eSpeak NG was not found. Replay will be disabled.\n");
        replay = 0;
    }

    printf("\nReady. Speak naturally; Vosk will finalize an utterance after silence.\n");
    printf("Press Q or ESC at any time to quit.\n\n");

    while (running) {
        int read_status;
        int accepted;
        const char *json;

        /* Non-blocking keyboard poll; audio reads are at most about 250 ms. */
        if (_kbhit()) {
            int key = _getch();
            if (key == 'q' || key == 'Q' || key == 27) {
                running = 0;
                break;
            }
        }

        read_status = zenith_audio_read(&audio, samples,
                                        ZENITH_FRAMES_PER_BUFFER,
                                        error, sizeof(error));
        if (read_status < 0) {
            fprintf(stderr, "\nAudio capture failed: %s\n", error);
            break;
        }
        if (read_status == 0) {
            fprintf(stderr, "\n[Audio input overflow; a small portion may have been lost.]\n");
            continue;
        }

        accepted = vosk.accept_waveform(
            recognizer,
            (const char *)samples,
            (int)(ZENITH_FRAMES_PER_BUFFER * sizeof(short))
        );

        if (accepted == 1) {
            json = vosk.result(recognizer);
            if (json) handle_utterance(json, &audio, voice, rate, replay);
            last_partial[0] = '\0';
        } else if (accepted == 0) {
            json = vosk.partial_result(recognizer);
            if (json) print_partial_if_changed(json, last_partial, sizeof(last_partial));
        } else {
            fprintf(stderr, "\nVosk reported an audio processing error.\n");
        }
    }

    /* Flush any speech that was in progress when Q/ESC was pressed. */
    if (recognizer) {
        const char *final_json = vosk.final_result(recognizer);
        if (final_json) {
            /* Do not replay during shutdown; still preserve final transcript. */
            handle_utterance(final_json, &audio, voice, rate, 0);
        }
    }

    printf("\n\nStopping Z.E.N.I.T.H....\n");
    zenith_audio_close(&audio);

    if (recognizer) vosk.recognizer_free(recognizer);
    if (model) vosk.model_free(model);
    zenith_vosk_unload(&vosk);

    printf("Session closed. Transcript: %s\n", ZENITH_TRANSCRIPT_PATH);
    return 0;
}
