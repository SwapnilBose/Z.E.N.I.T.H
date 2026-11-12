
#ifndef ZA_DATASET_H
#define ZA_DATASET_H

#include "za_matrix.h"
#include <stdint.h>

/*
 * A dataset contains:
 *
 * inputs:
 *     One row per sample, one column per input feature.
 *
 * targets:
 *     One row per sample, one column per expected output.
 */
typedef struct
{
    Matrix inputs;
    Matrix targets;

    size_t sample_count;
    size_t input_size;
    size_t output_size;

} Dataset;


/*
 * Load a numeric CSV dataset.
 *
 * Example CSV:
 * x1,x2,target
 * 1,2,5
 * 2,1,5
 *
 * input_size  = 2
 * output_size = 1
 * has_header  = 1
 *
 * Returns an empty Dataset if loading fails.
 */
Dataset dataset_load_csv(
    const char *filename,
    size_t input_size,
    size_t output_size,
    int has_header
);


/*
 * Randomly rearrange dataset samples.
 *
 * Input and target rows are always moved together.
 * The same seed produces the same permutation.
 *
 * Returns 1 on success, 0 on failure.
 */
int dataset_shuffle(Dataset *dataset, uint64_t seed);

/*
 * Divide a dataset into training, validation, and testing sets.
 *
 * validation_ratio is the fraction reserved for validation.
 * The remaining fraction after training and validation
 * becomes the testing fraction.
 *
 * All three output Dataset objects must initially be empty:
 * {0}
 *
 * Returns 1 on success, 0 on failure.
 */
int dataset_split(
    const Dataset *dataset,
    float training_ratio,
    float validation_ratio,
    uint64_t seed,
    Dataset *training,
    Dataset *validation,
    Dataset *testing
);



/* Release all dataset memory. */
void dataset_free(Dataset *dataset);

#endif
