
#ifndef ZA_SOFTMAX_H
#define ZA_SOFTMAX_H

#include "za_matrix.h"

/*
 * Apply numerically stable softmax to a matrix.
 *
 * Every row represents one sample.
 * Every column represents one class.
 *
 * Example:
 *
 * Input:
 * [2, 1, 0]
 *
 * Output:
 * [0.6652, 0.2447, 0.0900]
 *
 * Each row of the output sums to approximately 1.
 *
 * Uses maximum subtraction to prevent exponential overflow.
 *
 * Returns:
 *     A newly allocated probability matrix on success.
 *     An empty matrix on failure.
 *
 * The caller owns the returned matrix and must call
 * matrix_free() when finished.
 */
Matrix softmax_forward(const Matrix *logits);

#endif
