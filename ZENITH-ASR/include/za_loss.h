
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


/*
 * Calculate the gradient of softmax followed by
 * categorical cross-entropy.
 *
 * Formula:
 *
 *     gradient = (probabilities - targets) / N
 *
 * N = number of samples.
 *
 * Every row represents one sample.
 * Every column represents one class.
 *
 * The returned matrix contains gradients with respect
 * to the original logits.
 *
 * Returns:
 *     A newly allocated gradient matrix on success.
 *     An empty matrix on failure.
 *
 * The caller owns the returned matrix.
 */
Matrix softmax_cross_entropy_gradient(
    const Matrix *probabilities,
    const Matrix *targets
);


#endif
