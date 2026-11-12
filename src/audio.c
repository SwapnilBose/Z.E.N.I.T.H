#include "zenith_audio.h"
#include "zenith_config.h"

#include <stdio.h>
#include <string.h>

static void set_error(char *out, unsigned long cap, const char *message)
{
    if (out && cap > 0) {
        snprintf(out, (size_t)cap, "%s", message ? message : "Unknown audio error");
    }
}

int zenith_audio_open(ZenithAudio *audio, char *error, unsigned long error_cap)
{
    PaError pa_error;
    PaDeviceIndex device_index;
    const PaDeviceInfo *device_info;
    PaStreamParameters input;
    const int rates[] = { ZENITH_DEFAULT_SAMPLE_RATE, ZENITH_FALLBACK_SAMPLE_RATE };
    size_t i;

    if (!audio) {
        set_error(error, error_cap, "Invalid audio object.");
        return 0;
    }

    memset(audio, 0, sizeof(*audio));

    pa_error = Pa_Initialize();
    if (pa_error != paNoError) {
        set_error(error, error_cap, Pa_GetErrorText(pa_error));
        return 0;
    }
    audio->initialized = 1;

    device_index = Pa_GetDefaultInputDevice();
    if (device_index == paNoDevice) {
        set_error(error, error_cap, "Windows reports no default microphone input device.");
        zenith_audio_close(audio);
        return 0;
    }

    device_info = Pa_GetDeviceInfo(device_index);
    if (!device_info || device_info->maxInputChannels < 1) {
        set_error(error, error_cap, "The default audio device has no input channel.");
        zenith_audio_close(audio);
        return 0;
    }

    memset(&input, 0, sizeof(input));
    input.device = device_index;
    input.channelCount = 1;
    input.sampleFormat = paInt16;
    input.suggestedLatency = device_info->defaultLowInputLatency;
    input.hostApiSpecificStreamInfo = NULL;

    for (i = 0; i < sizeof(rates) / sizeof(rates[0]); ++i) {
        pa_error = Pa_OpenStream(&audio->stream,
                                 &input,
                                 NULL,
                                 (double)rates[i],
                                 ZENITH_FRAMES_PER_BUFFER,
                                 paClipOff,
                                 NULL,
                                 NULL);
        if (pa_error == paNoError) {
            audio->sample_rate = rates[i];
            return 1;
        }
        audio->stream = NULL;
    }

    {
        char message[256];
        snprintf(message, sizeof(message),
                 "Could not open the default microphone at 16 kHz or 48 kHz. PortAudio: %s",
                 Pa_GetErrorText(pa_error));
        set_error(error, error_cap, message);
    }
    zenith_audio_close(audio);
    return 0;
}

int zenith_audio_start(ZenithAudio *audio, char *error, unsigned long error_cap)
{
    PaError pa_error;
    if (!audio || !audio->stream) {
        set_error(error, error_cap, "Audio stream is not open.");
        return 0;
    }
    if (audio->started) return 1;

    pa_error = Pa_StartStream(audio->stream);
    if (pa_error != paNoError) {
        set_error(error, error_cap, Pa_GetErrorText(pa_error));
        return 0;
    }
    audio->started = 1;
    return 1;
}

int zenith_audio_stop(ZenithAudio *audio, char *error, unsigned long error_cap)
{
    PaError pa_error;
    if (!audio || !audio->stream || !audio->started) return 1;

    pa_error = Pa_StopStream(audio->stream);
    if (pa_error != paNoError) {
        set_error(error, error_cap, Pa_GetErrorText(pa_error));
        return 0;
    }
    audio->started = 0;
    return 1;
}

int zenith_audio_read(ZenithAudio *audio, short *buffer, unsigned long frames,
                      char *error, unsigned long error_cap)
{
    PaError pa_error;
    if (!audio || !audio->stream || !audio->started || !buffer) {
        set_error(error, error_cap, "Audio stream is not running.");
        return -1;
    }

    pa_error = Pa_ReadStream(audio->stream, buffer, (unsigned long)frames);
    if (pa_error == paNoError) return 1;
    if (pa_error == paInputOverflowed) return 0;

    set_error(error, error_cap, Pa_GetErrorText(pa_error));
    return -1;
}

void zenith_audio_close(ZenithAudio *audio)
{
    if (!audio) return;

    if (audio->stream) {
        if (audio->started) {
            Pa_StopStream(audio->stream);
            audio->started = 0;
        }
        Pa_CloseStream(audio->stream);
        audio->stream = NULL;
    }

    if (audio->initialized) {
        Pa_Terminate();
        audio->initialized = 0;
    }
}
