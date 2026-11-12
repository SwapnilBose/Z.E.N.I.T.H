#include "zenith_json.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>

static int hex_value(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

static int read_u16(const char **cursor, unsigned int *value)
{
    unsigned int v = 0;
    int i;
    for (i = 0; i < 4; ++i) {
        int h = hex_value((*cursor)[i]);
        if (h < 0) return 0;
        v = (v << 4) | (unsigned int)h;
    }
    *cursor += 4;
    *value = v;
    return 1;
}

static int append_utf8(unsigned int cp, char *out, size_t cap, size_t *used)
{
    unsigned char bytes[4];
    size_t count;

    if (cp <= 0x7F) {
        bytes[0] = (unsigned char)cp; count = 1;
    } else if (cp <= 0x7FF) {
        bytes[0] = (unsigned char)(0xC0 | (cp >> 6));
        bytes[1] = (unsigned char)(0x80 | (cp & 0x3F)); count = 2;
    } else if (cp <= 0xFFFF) {
        bytes[0] = (unsigned char)(0xE0 | (cp >> 12));
        bytes[1] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
        bytes[2] = (unsigned char)(0x80 | (cp & 0x3F)); count = 3;
    } else if (cp <= 0x10FFFF) {
        bytes[0] = (unsigned char)(0xF0 | (cp >> 18));
        bytes[1] = (unsigned char)(0x80 | ((cp >> 12) & 0x3F));
        bytes[2] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
        bytes[3] = (unsigned char)(0x80 | (cp & 0x3F)); count = 4;
    } else {
        return 0;
    }

    if (*used + count >= cap) return 0;
    memcpy(out + *used, bytes, count);
    *used += count;
    return 1;
}

int zenith_json_get_string(const char *json, const char *key,
                           char *output, size_t output_size)
{
    char pattern[128];
    const char *p;
    size_t used = 0;

    if (!json || !key || !output || output_size == 0) return 0;
    output[0] = '\0';
    if (snprintf(pattern, sizeof(pattern), "\"%s\"", key) >= (int)sizeof(pattern))
        return 0;

    p = strstr(json, pattern);
    if (!p) return 0;
    p += strlen(pattern);

    while (*p && isspace((unsigned char)*p)) ++p;
    if (*p++ != ':') return 0;
    while (*p && isspace((unsigned char)*p)) ++p;
    if (*p++ != '"') return 0;

    while (*p && *p != '"') {
        unsigned int cp;
        unsigned char c = (unsigned char)*p++;

        if (c != '\\') {
            if (used + 1 >= output_size) break;
            output[used++] = (char)c;
            continue;
        }

        if (!*p) break;
        c = (unsigned char)*p++;
        switch (c) {
            case '"': cp = '"'; break;
            case '\\': cp = '\\'; break;
            case '/': cp = '/'; break;
            case 'b': cp = '\b'; break;
            case 'f': cp = '\f'; break;
            case 'n': cp = '\n'; break;
            case 'r': cp = '\r'; break;
            case 't': cp = '\t'; break;
            case 'u': {
                unsigned int first;
                if (!read_u16(&p, &first)) { output[used] = '\0'; return 0; }
                cp = first;
                /* Combine a UTF-16 surrogate pair, if present. */
                if (first >= 0xD800 && first <= 0xDBFF && p[0] == '\\' && p[1] == 'u') {
                    const char *low_cursor = p + 2;
                    unsigned int second;
                    if (read_u16(&low_cursor, &second) && second >= 0xDC00 && second <= 0xDFFF) {
                        cp = 0x10000 + (((first - 0xD800) << 10) | (second - 0xDC00));
                        p = low_cursor;
                    }
                }
                break;
            }
            default:
                cp = c;
                break;
        }

        if (cp <= 0x7F) {
            if (used + 1 >= output_size) break;
            output[used++] = (char)cp;
        } else if (!append_utf8(cp, output, output_size, &used)) {
            break;
        }
    }

    output[used] = '\0';
    return 1;
}
