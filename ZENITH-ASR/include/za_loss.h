
#ifndef ZA_LOSS_H
#define ZA_LOSS_H

#include "za_matrix.h"

/*
 * Calculate categorical cross-entropy loss.
 *
 * probabilities:
 *     Predicted class probabilities from softmax.
 *
 * targets:
 *     One-hot encoded expected class probabilities.
 *
 * Both matrices must have identical dimensions:
 *
 *     rows    = number of samples
 *     columns = number of classes
 *
 * Each target row must sum to approximately 1.
 * Each probability row must sum to approximately 1.
 *
 * The function calculates the mean loss across all samples.
 *
 * A small epsilon is used to prevent log(0).
 *
 * Returns:
 *     Mean cross-entropy loss on success.
 *     NAN on invalid input.
 */
float cross_entropy_loss(
    const Matrix *probabilities,
    const Matrix *targets
);

#endif
