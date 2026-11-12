#ifndef ZENITH_VOSK_H
#define ZENITH_VOSK_H

#include <windows.h>

/* Opaque types from Vosk's public C API. Dynamic loading avoids requiring a
   compiler-specific Vosk import library (.a or .lib). */
typedef struct VoskModel VoskModel;
typedef struct VoskRecognizer VoskRecognizer;

typedef struct ZenithVosk {
    HMODULE dll;
    VoskModel *(*model_new)(const char *model_path);
    void (*model_free)(VoskModel *model);
    VoskRecognizer *(*recognizer_new)(VoskModel *model, float sample_rate);
    int (*accept_waveform)(VoskRecognizer *recognizer, const char *data, int length);
    const char *(*result)(VoskRecognizer *recognizer);
    const char *(*partial_result)(VoskRecognizer *recognizer);
    const char *(*final_result)(VoskRecognizer *recognizer);
    void (*recognizer_reset)(VoskRecognizer *recognizer);
    void (*recognizer_free)(VoskRecognizer *recognizer);
    void (*set_log_level)(int level);
} ZenithVosk;

int zenith_vosk_load(ZenithVosk *api, const char *dll_path,
                     char *error, unsigned long error_cap);
void zenith_vosk_unload(ZenithVosk *api);
VoskModel *zenith_vosk_model_new(ZenithVosk *api, const char *model_path);
VoskRecognizer *zenith_vosk_recognizer_new(ZenithVosk *api,
                                            VoskModel *model,
                                            float sample_rate);

#endif
