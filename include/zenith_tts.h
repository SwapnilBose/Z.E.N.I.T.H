#ifndef ZENITH_TTS_H
#define ZENITH_TTS_H

/* Uses the installed eSpeak NG command-line program. Returns 1 on success. */
int zenith_speak(const char *text, const char *voice, int words_per_minute);
int zenith_espeak_available(void);

#endif
