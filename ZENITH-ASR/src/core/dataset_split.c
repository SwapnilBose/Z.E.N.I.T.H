
#include "za_dataset.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/*
 * A small deterministic pseudo-random number generator.
 *
 * The state must never be zero.
 */
static uint64_t random_next(uint64_t *state)
{
    uint64_t x = *state;

    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;

    *state = x;

    return x * UINT64_C(2685821657736338717);
}

/*
 * Generate a random integer in [0, bound).
 *
 * Rejection sampling avoids modulo bias.
 */
static size_t random_bounded(uint64_t *state, size_t bound)
{
    uint64_t limit = (uint64_t)bound;
    uint64_t threshold = (uint64_t)(0 - limit) % limit;

    uint64_t value;

    do
    {
        value = random_next(state);
    }
    while (value < threshold);

    return (size_t)(value % limit);
}

/*
 * Initialize a random state.
 */
static uint64_t random_seed(uint64_t seed)
{
    if (seed == 0)
    {
        return UINT64_C(0x9E3779B97F4A7C15);
    }

    return seed;
}

/*
 * Swap two rows in a matrix.
 */
static void swap_matrix_rows(Matrix *matrix, size_t a, size_t b)
{
    if (a == b)
    {
        return;
    }

    for (size_t j = 0; j < matrix->cols; j++)
    {
        size_t index_a = a * matrix->cols + j;
        size_t index_b = b * matrix->cols + j;

        float temporary = matrix->data[index_a];

        matrix->data[index_a] = matrix->data[index_b];
        matrix->data[index_b] = temporary;
    }
}

/*
 * Check that a Dataset has matching input and target dimensions.
 */
static int dataset_is_valid(const Dataset *dataset)
{
    if (dataset == NULL ||
        dataset->inputs.data == NULL ||
        dataset->targets.data == NULL)
    {
        return 0;
    }

    if (dataset->sample_count == 0 ||
        dataset->input_size == 0 ||
        dataset->output_size == 0)
    {
        return 0;
    }

    if (dataset->inputs.rows != dataset->sample_count ||
        dataset->targets.rows != dataset->sample_count ||
        dataset->inputs.cols != dataset->input_size ||
        dataset->targets.cols != dataset->output_size)
    {
        return 0;
    }

    return 1;
}

/*
 * Check that an output Dataset is empty.
 */
static int dataset_is_empty(const Dataset *dataset)
{
    return dataset != NULL &&
           dataset->inputs.data == NULL &&
           dataset->targets.data == NULL &&
           dataset->sample_count == 0 &&
           dataset->input_size == 0 &&
           dataset->output_size == 0;
}

/*
 * Copy one sample from a source dataset into a destination.
 */
static void copy_sample(
    const Dataset *source,
    Dataset *destination,
    size_t source_row,
    size_t destination_row
)
{
    for (size_t j = 0; j < source->input_size; j++)
    {
        destination->inputs.data[
            destination_row * destination->input_size + j
        ] = source->inputs.data[
            source_row * source->input_size + j
        ];
    }

    for (size_t j = 0; j < source->output_size; j++)
    {
        destination->targets.data[
            destination_row * destination->output_size + j
        ] = source->targets.data[
            source_row * source->output_size + j
        ];
    }
}

/*
 * Allocate a dataset with the requested number of samples.
 */
static int dataset_allocate(
    Dataset *dataset,
    size_t sample_count,
    size_t input_size,
    size_t output_size
)
{
    dataset->inputs = matrix_create(sample_count, input_size);

    if (dataset->inputs.data == NULL)
    {
        return 0;
    }

    dataset->targets = matrix_create(sample_count, output_size);

    if (dataset->targets.data == NULL)
    {
        dataset_free(dataset);
        return 0;
    }

    dataset->sample_count = sample_count;
    dataset->input_size = input_size;
    dataset->output_size = output_size;

    return 1;
}

/*
 * Randomly rearrange a dataset using Fisher-Yates.
 */
int dataset_shuffle(Dataset *dataset, uint64_t seed)
{
    if (!dataset_is_valid(dataset))
    {
        return 0;
    }

    uint64_t state = random_seed(seed);

    /*
     * Fisher-Yates:
     * For each position, swap it with a randomly
     * selected position from the remaining prefix.
     */
    for (size_t i = dataset->sample_count - 1; i > 0; i--)
    {
        size_t j = random_bounded(&state, i + 1);

        swap_matrix_rows(&dataset->inputs, i, j);
        swap_matrix_rows(&dataset->targets, i, j);
    }

    return 1;
}

/*
 * Split a dataset into three independent datasets.
 */
int dataset_split(
    const Dataset *dataset,
    float training_ratio,
    float validation_ratio,
    uint64_t seed,
    Dataset *training,
    Dataset *validation,
    Dataset *testing
)
{
    if (!dataset_is_valid(dataset) ||
        !dataset_is_empty(training) ||
        !dataset_is_empty(validation) ||
        !dataset_is_empty(testing))
    {
        return 0;
    }

    /*
     * All three output pointers must be distinct.
     */
    if (training == validation ||
        training == testing ||
        validation == testing)
    {
        return 0;
    }

    /*
     * Validate split ratios.
     */
    if (!isfinite(training_ratio) ||
        !isfinite(validation_ratio) ||
        training_ratio <= 0.0f ||
        validation_ratio <= 0.0f ||
        (double)training_ratio + validation_ratio >= 1.0)
    {
        return 0;
    }

    size_t total = dataset->sample_count;

    /*
     * Calculate split sizes.
     * The test set receives all remaining samples.
     */
    size_t training_count =
        (size_t)((double)total * training_ratio);

    size_t validation_count =
        (size_t)((double)total * validation_ratio);

    size_t testing_count =
        total - training_count - validation_count;

    /*
     * Every subset must contain at least one sample.
     */
    if (training_count == 0 ||
        validation_count == 0 ||
        testing_count == 0)
    {
        return 0;
    }

    /*
     * Build an array of sample indices.
     */
    if (total > SIZE_MAX / sizeof(size_t))
    {
        return 0;
    }

    size_t *indices = malloc(total * sizeof(size_t));

    if (indices == NULL)
    {
        return 0;
    }

    for (size_t i = 0; i < total; i++)
    {
        indices[i] = i;
    }

    /*
     * Shuffle the indices, not the original dataset.
     * This preserves the source dataset.
     */
    uint64_t state = random_seed(seed);

    for (size_t i = total - 1; i > 0; i--)
    {
        size_t j = random_bounded(&state, i + 1);

        size_t temporary = indices[i];
        indices[i] = indices[j];
        indices[j] = temporary;
    }

    /*
     * Allocate temporary output datasets.
     * We only publish them after all operations succeed.
     */
    Dataset train_temp = {0};
    Dataset validation_temp = {0};
    Dataset test_temp = {0};

    int success = 0;

    if (!dataset_allocate(
            &train_temp,
            training_count,
            dataset->input_size,
            dataset->output_size))
    {
        goto cleanup;
    }

    if (!dataset_allocate(
            &validation_temp,
            validation_count,
            dataset->input_size,
            dataset->output_size))
    {
        goto cleanup;
    }

    if (!dataset_allocate(
            &test_temp,
            testing_count,
            dataset->input_size,
            dataset->output_size))
    {
        goto cleanup;
    }

    /*
     * Copy shuffled samples into their respective sets.
     */
    for (size_t i = 0; i < training_count; i++)
    {
        copy_sample(
            dataset,
            &train_temp,
            indices[i],
            i
        );
    }

    for (size_t i = 0; i < validation_count; i++)
    {
        copy_sample(
            dataset,
            &validation_temp,
            indices[training_count + i],
            i
        );
    }

    for (size_t i = 0; i < testing_count; i++)
    {
        copy_sample(
            dataset,
            &test_temp,
            indices[training_count + validation_count + i],
            i
        );
    }

    /*
     * Transfer ownership to the caller.
     */
    *training = train_temp;
    *validation = validation_temp;
    *testing = test_temp;

    success = 1;

cleanup:

    free(indices);

    if (!success)
    {
        dataset_free(&train_temp);
        dataset_free(&validation_temp);
        dataset_free(&test_temp);
    }

    return success;
}
