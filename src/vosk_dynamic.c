#include "zenith_vosk.h"

#include <stdio.h>
#include <string.h>

static void set_error(char *out, unsigned long cap, const char *message)
{
    if (out && cap > 0) {
        snprintf(out, (size_t)cap, "%s", message ? message : "Unknown Vosk error");
    }
}

#define LOAD_SYMBOL(api, field, symbol_name)                                      \
    do {                                                                           \
        FARPROC proc_address = GetProcAddress((api)->dll, (symbol_name));          \
        if (!proc_address) {                                                        \
            char msg[256];                                                         \
            snprintf(msg, sizeof(msg), "Vosk DLL is missing export: %s",         \
                     (symbol_name));                                               \
            set_error(error, error_cap, msg);                                      \
            zenith_vosk_unload(api);                                               \
            return 0;                                                              \
        }                                                                          \
        memcpy(&(api)->field, &proc_address, sizeof((api)->field));                \
    } while (0)

/* Windows function pointers and FARPROC have the same pointer width. */
int zenith_vosk_load(ZenithVosk *api, const char *dll_path,
                     char *error, unsigned long error_cap)
{
    if (!api || !dll_path) {
        set_error(error, error_cap, "Invalid Vosk loader arguments.");
        return 0;
    }

    memset(api, 0, sizeof(*api));

    /*
    * Convert the DLL path to an absolute Windows path.
    * LOAD_WITH_ALTERED_SEARCH_PATH works more reliably
    * when the DLL path is fully qualified.
    */
    char absolute_dll_path[MAX_PATH];

    DWORD path_length = GetFullPathNameA(
        dll_path,
        MAX_PATH,
        absolute_dll_path,
        NULL
    );

    if (path_length == 0 || path_length >= MAX_PATH) {
        DWORD code = GetLastError();

        char message[256];
        snprintf(message, sizeof(message),
                "Could not resolve Vosk DLL path. Windows error %lu",
                (unsigned long)code);

        set_error(error, error_cap, message);
        return 0;
    }

    /*
    * Load Vosk using its absolute path.
    * Windows can then locate companion DLLs in the same directory.
    */
    api->dll = LoadLibraryExA(
        absolute_dll_path,
        NULL,
        LOAD_WITH_ALTERED_SEARCH_PATH
    );
    if (!api->dll) {
        DWORD code = GetLastError();
        char message[256];
        snprintf(message, sizeof(message),
                 "Could not load '%s' (Windows error %lu). Copy libvosk.dll and its companion DLLs into bin/.",
                 dll_path, (unsigned long)code);
        set_error(error, error_cap, message);
        return 0;
    }

    LOAD_SYMBOL(api, model_new, "vosk_model_new");
    LOAD_SYMBOL(api, model_free, "vosk_model_free");
    LOAD_SYMBOL(api, recognizer_new, "vosk_recognizer_new");
    LOAD_SYMBOL(api, accept_waveform, "vosk_recognizer_accept_waveform");
    LOAD_SYMBOL(api, result, "vosk_recognizer_result");
    LOAD_SYMBOL(api, partial_result, "vosk_recognizer_partial_result");
    LOAD_SYMBOL(api, final_result, "vosk_recognizer_final_result");
    LOAD_SYMBOL(api, recognizer_reset, "vosk_recognizer_reset");
    LOAD_SYMBOL(api, recognizer_free, "vosk_recognizer_free");
    LOAD_SYMBOL(api, set_log_level, "vosk_set_log_level");

    api->set_log_level(-1);
    return 1;
}

void zenith_vosk_unload(ZenithVosk *api)
{
    if (!api) return;
    if (api->dll) {
        FreeLibrary(api->dll);
        api->dll = NULL;
    }
    memset(api, 0, sizeof(*api));
}

VoskModel *zenith_vosk_model_new(ZenithVosk *api, const char *model_path)
{
    if (!api || !api->model_new || !model_path) return NULL;
    return api->model_new(model_path);
}

VoskRecognizer *zenith_vosk_recognizer_new(ZenithVosk *api,
                                            VoskModel *model,
                                            float sample_rate)
{
    if (!api || !api->recognizer_new || !model) return NULL;
    return api->recognizer_new(model, sample_rate);
}
