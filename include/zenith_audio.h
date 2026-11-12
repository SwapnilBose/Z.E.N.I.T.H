#ifndef ZENITH_AUDIO_H
#define ZENITH_AUDIO_H

#include <portaudio.h>

typedef struct ZenithAudio {
    PaStream *stream;
    int sample_rate;
    int initialized;
    int started;
} ZenithAudio;

/* Initializes PortAudio and opens the default mono input device.
   Tries 16 kHz first, then 48 kHz; returns the selected sample rate. */
int zenith_audio_open(ZenithAudio *audio, char *error, unsigned long error_cap);

int zenith_audio_start(ZenithAudio *audio, char *error, unsigned long error_cap);
int zenith_audio_stop(ZenithAudio *audio, char *error, unsigned long error_cap);

/* Reads signed 16-bit mono PCM samples into caller-owned buffer.
   Returns 1 on success, 0 for a recoverable input overflow, -1 on fatal error. */
int zenith_audio_read(ZenithAudio *audio, short *buffer, unsigned long frames,
                      char *error, unsigned long error_cap);

void zenith_audio_close(ZenithAudio *audio);

#endif
