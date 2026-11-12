
#include "za_dataset.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CSV_LINE_SIZE 8192

/*
 * Parse one CSV row.
 *
 * Returns:
 *   1  = valid row
 *   0  = blank row
 *  -1  = invalid row
 *
 * This initial loader supports numeric CSV files only.
 * Quoted CSV fields are intentionally not supported.
 */
static int parse_csv_row(
    char *line,
    float *row,
    size_t column_count)
{
    char *position = line;

    /* Ignore leading whitespace. */
    while (isspace((unsigned char)*position))
        position++;

    /* Ignore completely blank lines. */
    if (*position == '\0')
        return 0;

    for (size_t column = 0;
         column < column_count;
         column++)
    {
        char *end = NULL;

        while (isspace((unsigned char)*position))
            position++;

        /* Reject missing values. */
        if (*position == '\0' || *position == ',')
            return -1;

        errno = 0;

        float value = strtof(position, &end);

        if (end == position ||
            errno == ERANGE ||
            !isfinite(value))
        {
            return -1;
        }

        row[column] = value;
        position = end;

        while (isspace((unsigned char)*position))
            position++;

        if (column + 1 < column_count)
        {
            if (*position != ',')
                return -1;

            position++;
        }
        else
        {
            /* No extra columns or trailing characters. */
            if (*position != '\0' &&
                *position != '\n' &&
                *position != '\r')
            {
                return -1;
            }
        }
    }

    return 1;
}


/*
 * Release a partially constructed dataset.
 */
void dataset_free(Dataset *dataset)
{
    if (dataset == NULL)
        return;

    matrix_free(&dataset->inputs);
    matrix_free(&dataset->targets);

    dataset->sample_count = 0;
    dataset->input_size = 0;
    dataset->output_size = 0;
}


/*
 * Load a CSV file into two matrices.
 *
 * CSV column arrangement:
 *
 * input_1,input_2,...,target_1,target_2,...
 *
 * The loader stores input and target values separately.
 */
Dataset dataset_load_csv(
    const char *filename,
    size_t input_size,
    size_t output_size,
    int has_header)
{
    Dataset dataset = {0};

    FILE *file = NULL;

    float *all_values = NULL;
    float *row = NULL;

    size_t sample_count = 0;
    size_t capacity = 0;
    size_t column_count;

    char line[CSV_LINE_SIZE];

    if (filename == NULL ||
        input_size == 0 ||
        output_size == 0)
    {
        return dataset;
    }

    /* Prevent column-count overflow. */
    if (input_size > SIZE_MAX - output_size)
        return dataset;

    column_count = input_size + output_size;

    if (column_count > SIZE_MAX / sizeof(float))
        return dataset;

    file = fopen(filename, "r");

    if (file == NULL)
        goto cleanup;

    /*
     * Optionally skip the first CSV line.
     * Its contents are treated as column names.
     */
    if (has_header)
    {
        if (fgets(line, sizeof(line), file) == NULL)
            goto cleanup;

        /* Reject a header longer than our line buffer. */
        if (strchr(line, '\n') == NULL && !feof(file))
            goto cleanup;
    }

    row = malloc(column_count * sizeof(float));

    if (row == NULL)
        goto cleanup;

    while (fgets(line, sizeof(line), file) != NULL)
    {
        int parse_result;

        /*
         * Detect lines that do not fit in the buffer.
         * This prevents silently parsing truncated data.
         */
        if (strchr(line, '\n') == NULL && !feof(file))
            goto cleanup;

        parse_result = parse_csv_row(
            line,
            row,
            column_count
        );

        if (parse_result < 0)
            goto cleanup;

        if (parse_result == 0)
            continue;

        /*
         * Increase storage when the current capacity
         * is exhausted.
         */
        if (sample_count == capacity)
        {
            size_t new_capacity =
                capacity == 0 ? 16 : capacity * 2;

            if (new_capacity < capacity ||
                new_capacity >
                    SIZE_MAX / column_count / sizeof(float))
            {
                goto cleanup;
            }

            float *new_values = realloc(
                all_values,
                new_capacity * column_count * sizeof(float)
            );

            if (new_values == NULL)
                goto cleanup;

            all_values = new_values;
            capacity = new_capacity;
        }

        memcpy(
            &all_values[sample_count * column_count],
            row,
            column_count * sizeof(float)
        );

        sample_count++;
    }

    if (ferror(file) || sample_count == 0)
        goto cleanup;

    /*
     * Allocate final input and target matrices.
     */
    dataset.inputs = matrix_create(
        sample_count,
        input_size
    );

    dataset.targets = matrix_create(
        sample_count,
        output_size
    );

    if (dataset.inputs.data == NULL ||
        dataset.targets.data == NULL)
    {
        goto cleanup;
    }

    /*
     * Separate the CSV columns into X and Y.
     */
    for (size_t sample = 0;
         sample < sample_count;
         sample++)
    {
        const float *source =
            &all_values[sample * column_count];

        memcpy(
            &dataset.inputs.data[sample * input_size],
            source,
            input_size * sizeof(float)
        );

        memcpy(
            &dataset.targets.data[sample * output_size],
            source + input_size,
            output_size * sizeof(float)
        );
    }

    dataset.sample_count = sample_count;
    dataset.input_size = input_size;
    dataset.output_size = output_size;

cleanup:

    if (file != NULL)
        fclose(file);

    free(all_values);
    free(row);

    /*
     * If any operation failed, release everything.
     */
    if (dataset.inputs.data == NULL ||
        dataset.targets.data == NULL)
    {
        dataset_free(&dataset);
    }

    return dataset;
}
