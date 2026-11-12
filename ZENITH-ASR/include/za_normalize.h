
#ifndef ZA_NORMALIZE_H
#define ZA_NORMALIZE_H

#include "za_matrix.h"

/*
 * Stores the statistics needed to normalize input features.
 *
 * means[j]  = mean of feature j
 * scales[j] = standard deviation of feature j
 *
 * For a constant feature, scales[j] is set to 1.0f
 * to prevent division by zero.
 */
typedef struct
{
    float *means;
    float *scales;
    size_t feature_count;
} Normalizer;

/*
 * Calculate the mean and standard deviation of every
 * input feature (column).
 *
 * Returns a valid Normalizer on success.
 * Returns an empty Normalizer on failure.
 */
Normalizer normalizer_fit(const Matrix *inputs);

/*
 * Apply previously calculated normalization statistics.
 *
 * The input matrix is not modified.
 * The function allocates and stores a new matrix in outputs.
 *
 * outputs must point to an empty Matrix:
 * {0, 0, NULL}
 *
 * Returns 1 on success, 0 on failure.
 */
int normalizer_transform(
    const Normalizer *normalizer,
    const Matrix *inputs,
    Matrix *outputs
);

/* Release all memory owned by a Normalizer. */
void normalizer_free(Normalizer *normalizer);

#endif
