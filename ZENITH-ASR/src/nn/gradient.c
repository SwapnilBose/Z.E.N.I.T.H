
#include "za_gradient.h"

#include <math.h>
#include <stdint.h>
#include <float.h>

/*
 * Validate a gradient matrix.
 *
 * A valid gradient must:
 *     - Have allocated data.
 *     - Have non-zero dimensions.
 *     - Have a safe element count.
 *     - Contain only finite values.
 */
static int validate_gradient(const Matrix *gradient)
{
    if (!gradient || !gradient->data)
        return 0;

    if (gradient->rows == 0 || gradient->cols == 0)
        return 0;

    /* Prevent multiplication overflow. */
    if (gradient->rows > SIZE_MAX / gradient->cols)
        return 0;

    size_t count = gradient->rows * gradient->cols;

    for (size_t i = 0; i < count; i++)
    {
        if (!isfinite(gradient->data[i]))
            return 0;
    }

    return 1;
}


/*
 * Calculate the global L2 norm.
 *
 * Double precision is used for accumulation
 * to reduce numerical errors.
 */
double gradient_global_norm(const Matrix *gradient)
{
    if (!validate_gradient(gradient))
        return NAN;

    size_t count = gradient->rows * gradient->cols;

    double squared_sum = 0.0;

    for (size_t i = 0; i < count; i++)
    {
        double value = (double)gradient->data[i];

        squared_sum += value * value;

        if (!isfinite(squared_sum))
            return NAN;
    }

    return sqrt(squared_sum);
}


/*
 * Clip the global gradient norm.
 *
 * All gradient elements are scaled by the same
 * factor, preserving their relative proportions.
 */
int gradient_clip_global_norm(
    Matrix *gradient,
    float max_norm)
{
    /* Validate clipping threshold. */
    if (!isfinite(max_norm) || max_norm <= 0.0f)
        return 0;

    /* Validate gradient data. */
    if (!validate_gradient(gradient))
        return 0;

    double norm = gradient_global_norm(gradient);

    if (!isfinite(norm))
        return 0;

    /*
     * No clipping is necessary when the norm
     * is already within the permitted limit.
     */
    if (norm <= max_norm)
        return 1;

    /*
     * Calculate scaling factor.
     */
    double scale =
        (double)max_norm / norm;

    if (!isfinite(scale))
        return 0;

    size_t count = gradient->rows * gradient->cols;

    /*
     * First calculate and validate all candidate
     * values. Do not modify the matrix yet.
     */
    for (size_t i = 0; i < count; i++)
    {
        double candidate =
            (double)gradient->data[i] * scale;

        if (!isfinite(candidate) ||
            fabs(candidate) > (double)FLT_MAX)
        {
            return 0;
        }
    }

    /*
     * All candidates are valid.
     * Now apply the clipping operation.
     */
    for (size_t i = 0; i < count; i++)
    {
        gradient->data[i] =
            (float)((double)gradient->data[i] * scale);
    }

    return 1;
}
