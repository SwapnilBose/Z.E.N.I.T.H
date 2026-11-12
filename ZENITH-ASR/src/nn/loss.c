
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
 * This implementation also supports soft targets,
 * provided each target row sums to 1.
 */
float cross_entropy_loss(
    const Matrix *probabilities,
    const Matrix *targets)
{
    /*
     * Validate matrix pointers and dimensions.
     */
    if (!probabilities || !targets)
        return NAN;

    if (!probabilities->data || !targets->data)
        return NAN;

    if (probabilities->rows == 0 ||
        probabilities->cols == 0)
        return NAN;

    /*
     * Both matrices must have identical dimensions.
     */
    if (probabilities->rows != targets->rows ||
        probabilities->cols != targets->cols)
        return NAN;

    /*
     * Validate both distributions.
     */
    if (!validate_distribution(probabilities))
        return NAN;

    if (!validate_distribution(targets))
        return NAN;

    /*
     * Accumulate loss using double precision.
     */
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
             *
             * This also prevents an infinite loss when
             * a model assigns exactly zero probability
             * to the correct class.
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

    /*
     * Return mean loss across all samples.
     */
    double mean_loss = total_loss / (double)probabilities->rows;

    if (!isfinite(mean_loss))
        return NAN;

    return (float)mean_loss;
}
