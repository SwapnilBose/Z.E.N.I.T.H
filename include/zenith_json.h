#ifndef ZENITH_JSON_H
#define ZENITH_JSON_H
#include <stddef.h>

/* Extracts a JSON string property and decodes common escapes and \uXXXX.
   Returns 1 when the property is found, 0 otherwise. */
int zenith_json_get_string(const char *json, const char *key,
                           char *output, size_t output_size);

#endif
