#include "zenith_tts.h"

#include <process.h>
#include <stdio.h>
#include <stdlib.h>

int zenith_espeak_available(void)
{
    const char *argv[] = { "espeak-ng.exe", "--version", NULL };
    intptr_t result = _spawnvp(_P_WAIT, "espeak-ng.exe", argv);
    return result != -1;
}

int zenith_speak(const char *text, const char *voice, int words_per_minute)
{
    char rate_text[16];
    const char *argv[8];
    intptr_t result;

    if (!text || !text[0]) return 1;
    if (!voice || !voice[0]) voice = "en-us";
    if (words_per_minute < 80 || words_per_minute > 450)
        words_per_minute = 165;

    snprintf(rate_text, sizeof(rate_text), "%d", words_per_minute);

    argv[0] = "espeak-ng.exe";
    argv[1] = "-v";
    argv[2] = voice;
    argv[3] = "-s";
    argv[4] = rate_text;
    argv[5] = text;
    argv[6] = NULL;

    /* _spawnvp waits for eSpeak to finish. Windows argument quoting is
       performed by the C runtime; the transcript is passed as one argument. */
    result = _spawnvp(_P_WAIT, "espeak-ng.exe", argv);
    if (result == -1) {
        fprintf(stderr,
                "[Z.E.N.I.T.H.] Could not launch espeak-ng.exe. Install eSpeak NG or add it to PATH.\n");
        return 0;
    }
    return result == 0;
}
