
#ifndef ZA_GRADIENT_H
#define ZA_GRADIENT_H

#include "za_matrix.h"

/*
 * Calculate the global L2 norm of a gradient matrix.
 *
 * Formula:
 *
 *     norm = sqrt(sum(gradient[i] * gradient[i]))
 *
 * Returns:
 *     L2 norm on success.
 *     NAN if the matrix is invalid or contains
 *     non-finite values.
 */
double gradient_global_norm(const Matrix *gradient);


/*
 * Apply global norm clipping to a gradient matrix.
 *
 * Formula:
 *
 *     scale = min(1, max_norm / global_norm)
 *
 *     gradient[i] *= scale
 *
 * Parameters:
 *     gradient  - Gradient matrix to modify.
 *     max_norm  - Maximum permitted global norm.
 *
 * Returns:
 *     1 on success.
 *     0 on invalid input or numerical failure.
 *
 * The original gradient is modified in place.
 */
int gradient_clip_global_norm(
    Matrix *gradient,
    float max_norm
);

#endif
