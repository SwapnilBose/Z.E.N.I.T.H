#ifndef ZENITH_TRANSCRIPT_H
#define ZENITH_TRANSCRIPT_H

/* Appends a UTF-8 transcript line with a local timestamp. */
int zenith_transcript_append(const char *text, const char *file_path);

#endif
