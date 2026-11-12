#include "zenith_transcript.h"

#include <stdio.h>
#include <time.h>

int zenith_transcript_append(const char *text, const char *file_path)
{
    FILE *file;
    time_t now;
    struct tm local_time;
    struct tm *local_ptr;
    char stamp[32];

    if (!text || !file_path) return 0;

    file = fopen(file_path, "ab");
    if (!file) return 0;

    now = time(NULL);
    local_ptr = localtime(&now);
    if (local_ptr) local_time = *local_ptr;
    else { local_time.tm_year = 70; local_time.tm_mon = 0; local_time.tm_mday = 1;
           local_time.tm_hour = 0; local_time.tm_min = 0; local_time.tm_sec = 0; }
    if (strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &local_time) == 0)
        snprintf(stamp, sizeof(stamp), "unknown-time");

    fprintf(file, "[%s] %s\r\n", stamp, text);
    fclose(file);
    return 1;
}
