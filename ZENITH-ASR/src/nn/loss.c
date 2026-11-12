
#include "za_loss.h"

#include <math.h>
#include <stdint.h>

/*
 * Numerical safety threshold.
 *
 * log(0) is negative infinity.
 * Clamping very small probabilities prevents this.
 */
#define LOSS_EPSILON 1.0e-7

/*
 * Allowed floating-point error when checking
 * whether a probability distribution sums to 1.
 */
#define SUM_TOLERANCE 1.0e-4


/*
 * Validate a matrix of probability distributions.
 *
 * Every element must:
 *     - Be finite.
 *     - Be between 0 and 1.
 *
 * Every row must sum approximately to 1.
 */
static int validate_distribution(const Matrix *matrix)
{
    if (!matrix || !matrix->data)
        return 0;

    if (matrix->rows == 0 || matrix->cols == 0)
        return 0;

    /* Prevent size_t multiplication overflow. */
    if (matrix->rows > SIZE_MAX / matrix->cols)
        return 0;

    for (size_t row = 0; row < matrix->rows; row++)
    {
        double sum = 0.0;

        for (size_t col = 0; col < matrix->cols; col++)
        {
            float value = matrix->data[row * matrix->cols + col];

            if (!isfinite(value))
                return 0;

            if (value < 0.0f || value > 1.0f)
                return 0;

            sum += (double)value;
        }

        if (fabs(sum - 1.0) > SUM_TOLERANCE)
            return 0;
    }

    return 1;
}


/*
 * Calculate categorical cross-entropy.
 *
 * Formula:
 *
 *     L = -(1/N) * sum_n sum_c y[n][c] * log(p[n][c])
 *
 * N = number of samples.
 * C = number of classes.
 *
 * Supports soft targets, provided each target
 * row sums to 1.
 */
float cross_entropy_loss(
    const Matrix *probabilities,
    const Matrix *targets)
{
    /* Validate matrix pointers. */
    if (!probabilities || !targets)
        return NAN;

    if (!probabilities->data || !targets->data)
        return NAN;

    if (probabilities->rows == 0 ||
        probabilities->cols == 0)
        return NAN;

    /* Both matrices must have identical dimensions. */
    if (probabilities->rows != targets->rows ||
        probabilities->cols != targets->cols)
        return NAN;

    /* Validate both probability distributions. */
    if (!validate_distribution(probabilities))
        return NAN;

    if (!validate_distribution(targets))
        return NAN;

    /* Accumulate loss using double precision. */
    double total_loss = 0.0;

    for (size_t row = 0; row < probabilities->rows; row++)
    {
        for (size_t col = 0; col < probabilities->cols; col++)
        {
            size_t index = row * probabilities->cols + col;

            double target = (double)targets->data[index];
            double probability = (double)probabilities->data[index];

            /*
             * Avoid log(0).
             */
            if (probability < LOSS_EPSILON)
                probability = LOSS_EPSILON;

            /*
             * Cross-entropy contribution:
             *
             *     -target * log(probability)
             */
            total_loss -= target * log(probability);
        }
    }

    /* Mean loss across all samples. */
    double mean_loss = total_loss / (double)probabilities->rows;

    if (!isfinite(mean_loss))
        return NAN;

    /* Return the loss BEFORE ending this function. */
    return (float)mean_loss;
}


/*
 * Calculate the gradient of combined softmax
 * and categorical cross-entropy.
 *
 * Mathematical result:
 *
 *     dL/dz = (probabilities - targets) / N
 *
 * This is the gradient with respect to the original
 * logits, not the probabilities.
 *
 * N is the number of samples in the batch.
 */
Matrix softmax_cross_entropy_gradient(
    const Matrix *probabilities,
    const Matrix *targets)
{
    Matrix gradient = {0};

    /* Validate pointers. */
    if (!probabilities || !targets)
        return gradient;

    if (!probabilities->data || !targets->data)
        return gradient;

    /* Validate dimensions. */
    if (probabilities->rows == 0 ||
        probabilities->cols == 0)
        return gradient;

    if (probabilities->rows != targets->rows ||
        probabilities->cols != targets->cols)
        return gradient;

    /* Reuse probability distribution validation. */
    if (!validate_distribution(probabilities))
        return gradient;

    if (!validate_distribution(targets))
        return gradient;

    /* Allocate the gradient matrix. */
    gradient = matrix_create(
        probabilities->rows,
        probabilities->cols
    );

    if (!gradient.data)
        return (Matrix){0};

    /*
     * Calculate the gradient for every sample
     * and every class.
     */
    size_t element_count =
        probabilities->rows * probabilities->cols;

    double sample_count = (double)probabilities->rows;

    for (size_t i = 0; i < element_count; i++)
    {
        double predicted = (double)probabilities->data[i];
        double expected = (double)targets->data[i];

        /*
         * Average gradients across the batch.
         */
        double value = (predicted - expected) / sample_count;

        if (!isfinite(value))
        {
            matrix_free(&gradient);
            return (Matrix){0};
        }

        gradient.data[i] = (float)value;
    }

    return gradient;
}
