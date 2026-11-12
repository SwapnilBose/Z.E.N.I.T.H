
#include "za_softmax.h"

#include <math.h>
#include <stdint.h>

/*
 * Numerically stable softmax implementation.
 *
 * The input matrix has:
 *
 *     rows    = number of samples
 *     columns = number of classes
 *
 * Softmax is applied independently to each row.
 */
Matrix softmax_forward(const Matrix *logits)
{
    Matrix probabilities = {0};

    /*
     * Validate the input pointer and matrix dimensions.
     */
    if (!logits || !logits->data)
        return probabilities;

    if (logits->rows == 0 || logits->cols == 0)
        return probabilities;

    /*
     * Prevent overflow when calculating the number
     * of matrix elements.
     */
    if (logits->rows > SIZE_MAX / logits->cols)
        return probabilities;

    /*
     * Check every input value.
     *
     * NaN and infinity would make the probability
     * calculation unreliable.
     */
    size_t element_count = logits->rows * logits->cols;

    for (size_t i = 0; i < element_count; i++)
    {
        if (!isfinite(logits->data[i]))
            return probabilities;
    }

    /*
     * Allocate the output matrix.
     *
     * It has the same dimensions as the input.
     */
    probabilities = matrix_create(logits->rows, logits->cols);

    if (!probabilities.data)
        return (Matrix){0};

    /*
     * Process each sample independently.
     */
    for (size_t row = 0; row < logits->rows; row++)
    {
        size_t row_start = row * logits->cols;

        /*
         * STEP 1:
         * Find the largest logit in this row.
         */
        float maximum = logits->data[row_start];

        for (size_t col = 1; col < logits->cols; col++)
        {
            float value = logits->data[row_start + col];

            if (value > maximum)
                maximum = value;
        }

        /*
         * STEP 2:
         * Calculate exponentials after subtracting
         * the maximum value.
         *
         * Use double for intermediate calculations
         * to improve numerical precision.
         */
        double exponential_sum = 0.0;

        for (size_t col = 0; col < logits->cols; col++)
        {
            size_t index = row_start + col;

            double shifted_logit =
                (double)logits->data[index] - (double)maximum;

            double exponential = exp(shifted_logit);

            probabilities.data[index] = (float)exponential;

            exponential_sum += exponential;
        }

        /*
         * STEP 3:
         * Normalize every exponential.
         *
         * The resulting probabilities sum to 1.
         */
        if (!isfinite(exponential_sum) || exponential_sum <= 0.0)
        {
            matrix_free(&probabilities);
            return (Matrix){0};
        }

        for (size_t col = 0; col < logits->cols; col++)
        {
            size_t index = row_start + col;

            probabilities.data[index] =
                (float)((double)probabilities.data[index]
                        / exponential_sum);
        }
    }

    return probabilities;
}
