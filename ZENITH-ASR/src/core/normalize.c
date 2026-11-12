
#include "za_normalize.h"

#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <stdint.h>
#include <string.h>

/*
 * Check whether a matrix has valid dimensions,
 * allocated storage, and finite values.
 */
static int matrix_values_are_finite(const Matrix *matrix)
{
    if (matrix == NULL ||
        matrix->data == NULL ||
        matrix->rows == 0 ||
        matrix->cols == 0)
    {
        return 0;
    }

    /* Prevent integer overflow during size calculation. */
    if (matrix->rows > SIZE_MAX / matrix->cols)
    {
        return 0;
    }

    size_t total = matrix->rows * matrix->cols;

    for (size_t i = 0; i < total; i++)
    {
        if (!isfinite(matrix->data[i]))
        {
            return 0;
        }
    }

    return 1;
}

/*
 * Calculate normalization statistics from a dataset.
 *
 * Each column represents one feature.
 */
Normalizer normalizer_fit(const Matrix *inputs)
{
    Normalizer normalizer = {0};

    if (!matrix_values_are_finite(inputs))
    {
        return normalizer;
    }

    size_t feature_count = inputs->cols;
    size_t sample_count = inputs->rows;

    normalizer.means = calloc(feature_count, sizeof(float));
    normalizer.scales = calloc(feature_count, sizeof(float));

    if (normalizer.means == NULL || normalizer.scales == NULL)
    {
        normalizer_free(&normalizer);
        return normalizer;
    }

    normalizer.feature_count = feature_count;

    /*
     * Calculate the mean of each feature.
     *
     * We use double for the sum to improve precision.
     */
    for (size_t j = 0; j < feature_count; j++)
    {
        double sum = 0.0;

        for (size_t i = 0; i < sample_count; i++)
        {
            size_t index = i * feature_count + j;
            sum += inputs->data[index];
        }

        double mean = sum / (double)sample_count;

        if (!isfinite(mean) || fabs(mean) > FLT_MAX)
        {
            normalizer_free(&normalizer);
            return normalizer;
        }

        normalizer.means[j] = (float)mean;
    }

    /*
     * Calculate population standard deviation.
     *
     * Variance = sum((x - mean)^2) / N
     * Standard deviation = sqrt(variance)
     */
    for (size_t j = 0; j < feature_count; j++)
    {
        double squared_difference_sum = 0.0;
        double mean = normalizer.means[j];

        for (size_t i = 0; i < sample_count; i++)
        {
            size_t index = i * feature_count + j;

            double difference =
                (double)inputs->data[index] - mean;

            squared_difference_sum +=
                difference * difference;
        }

        double variance =
            squared_difference_sum / (double)sample_count;

        double standard_deviation = sqrt(variance);

        if (!isfinite(standard_deviation) || standard_deviation > FLT_MAX)
        {
            normalizer_free(&normalizer);
            return normalizer;
        }

        /*
         * Constant feature:
         * avoid division by zero by using a scale of 1.
         */
        if (standard_deviation == 0.0)
        {
            standard_deviation = 1.0;
        }

        normalizer.scales[j] = (float)standard_deviation;
    }

    return normalizer;
}

/*
 * Apply a fitted Normalizer to an input matrix.
 *
 * This creates a new output matrix.
 * The original input matrix remains unchanged.
 */
int normalizer_transform(
    const Normalizer *normalizer,
    const Matrix *inputs,
    Matrix *outputs
)
{
    if (normalizer == NULL ||
        inputs == NULL ||
        outputs == NULL)
    {
        return 0;
    }

    if (normalizer->means == NULL ||
        normalizer->scales == NULL ||
        normalizer->feature_count == 0)
    {
        return 0;
    }

    if (!matrix_values_are_finite(inputs))
    {
        return 0;
    }

    if (inputs->cols != normalizer->feature_count)
    {
        return 0;
    }

    /*
     * Require an empty output matrix so we don't
     * accidentally overwrite an existing allocation.
     */
    if (outputs->data != NULL)
    {
        return 0;
    }

    Matrix result = matrix_create(inputs->rows, inputs->cols);

    if (result.data == NULL)
    {
        return 0;
    }

    for (size_t i = 0; i < inputs->rows; i++)
    {
        for (size_t j = 0; j < inputs->cols; j++)
        {
            size_t index = i * inputs->cols + j;

            double value = inputs->data[index];
            double mean = normalizer->means[j];
            double scale = normalizer->scales[j];

            if (!isfinite(mean) ||
                !isfinite(scale) ||
                scale <= 0.0)
            {
                matrix_free(&result);
                return 0;
            }

            double normalized = (value - mean) / scale;

            if (!isfinite(normalized) ||
                fabs(normalized) > FLT_MAX)
            {
                matrix_free(&result);
                return 0;
            }

            result.data[index] = (float)normalized;
        }
    }

    *outputs = result;
    return 1;
}

/*
 * Free the statistics and reset the structure.
 */
void normalizer_free(Normalizer *normalizer)
{
    if (normalizer == NULL)
    {
        return;
    }

    free(normalizer->means);
    free(normalizer->scales);

    normalizer->means = NULL;
    normalizer->scales = NULL;
    normalizer->feature_count = 0;
}
