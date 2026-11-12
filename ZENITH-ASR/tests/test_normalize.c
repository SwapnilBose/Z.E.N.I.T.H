
#include "za_dataset.h"
#include "za_normalize.h"

#include <math.h>
#include <stdio.h>

#define TOLERANCE 0.001f

static int test_dataset_normalization(void)
{
    printf("Test 1: Dataset normalization\n");
    printf("-----------------------------\n");

    Dataset dataset = dataset_load_csv(
        "data/linear.csv",
        2,
        1,
        1
    );

    if (dataset.inputs.data == NULL)
    {
        printf("FAILED: Could not load dataset.\n");
        return 0;
    }

    Normalizer normalizer = normalizer_fit(&dataset.inputs);

    if (normalizer.means == NULL)
    {
        printf("FAILED: Could not fit normalizer.\n");
        dataset_free(&dataset);
        return 0;
    }

    Matrix normalized = {0};

    if (!normalizer_transform(
            &normalizer,
            &dataset.inputs,
            &normalized))
    {
        printf("FAILED: Could not transform dataset.\n");

        normalizer_free(&normalizer);
        dataset_free(&dataset);
        return 0;
    }

    printf("Feature statistics:\n");

    for (size_t j = 0; j < normalized.cols; j++)
    {
        double mean = 0.0;

        for (size_t i = 0; i < normalized.rows; i++)
        {
            mean += normalized.data[i * normalized.cols + j];
        }

        mean /= (double)normalized.rows;

        double variance = 0.0;

        for (size_t i = 0; i < normalized.rows; i++)
        {
            double difference =
                normalized.data[i * normalized.cols + j] - mean;

            variance += difference * difference;
        }

        variance /= (double)normalized.rows;

        double standard_deviation = sqrt(variance);

        printf(
            "Feature %zu: mean = %.6f, stddev = %.6f\n",
            j + 1,
            mean,
            standard_deviation
        );

        if (fabs(mean) > TOLERANCE ||
            fabs(standard_deviation - 1.0) > TOLERANCE)
        {
            printf("FAILED: Feature statistics are incorrect.\n");

            matrix_free(&normalized);
            normalizer_free(&normalizer);
            dataset_free(&dataset);
            return 0;
        }
    }

    /*
     * Check that the original dataset was not modified.
     *
     * Its first two values should still be 1 and 2.
     */
    if (dataset.inputs.data[0] != 1.0f ||
        dataset.inputs.data[1] != 2.0f)
    {
        printf("FAILED: Original dataset was modified.\n");

        matrix_free(&normalized);
        normalizer_free(&normalizer);
        dataset_free(&dataset);
        return 0;
    }

    printf("\nNormalized data:\n");

    for (size_t i = 0; i < normalized.rows; i++)
    {
        printf("[");

        for (size_t j = 0; j < normalized.cols; j++)
        {
            printf(
                "%8.4f",
                normalized.data[i * normalized.cols + j]
            );
        }

        printf(" ]\n");
    }

    matrix_free(&normalized);
    normalizer_free(&normalizer);
    dataset_free(&dataset);

    printf("Dataset normalization PASSED.\n\n");
    return 1;
}

static int test_constant_feature(void)
{
    printf("Test 2: Constant feature handling\n");
    printf("---------------------------------\n");

    Matrix inputs = matrix_create(3, 2);

    if (inputs.data == NULL)
    {
        printf("FAILED: Could not create test matrix.\n");
        return 0;
    }

    /*
     * Column 1 varies.
     * Column 2 is always 5.
     */
    float values[] = {
        1.0f, 5.0f,
        2.0f, 5.0f,
        3.0f, 5.0f
    };

    for (size_t i = 0; i < 6; i++)
    {
        inputs.data[i] = values[i];
    }

    Normalizer normalizer = normalizer_fit(&inputs);

    if (normalizer.means == NULL)
    {
        printf("FAILED: Could not fit constant-feature test.\n");
        matrix_free(&inputs);
        return 0;
    }

    Matrix normalized = {0};

    if (!normalizer_transform(&normalizer, &inputs, &normalized))
    {
        printf("FAILED: Constant-feature transformation failed.\n");

        normalizer_free(&normalizer);
        matrix_free(&inputs);
        return 0;
    }

    for (size_t i = 0; i < normalized.rows; i++)
    {
        float constant_value = normalized.data[i * 2 + 1];

        if (fabsf(constant_value) > TOLERANCE)
        {
            printf("FAILED: Constant feature was not normalized to zero.\n");

            matrix_free(&normalized);
            normalizer_free(&normalizer);
            matrix_free(&inputs);
            return 0;
        }
    }

    printf("Constant feature normalized to zero.\n");

    matrix_free(&normalized);
    normalizer_free(&normalizer);
    matrix_free(&inputs);

    printf("Constant feature test PASSED.\n\n");
    return 1;
}

int main(void)
{
    printf("ZENITH-ASR Normalization Tests\n");
    printf("==============================\n\n");

    if (!test_dataset_normalization())
    {
        return 1;
    }

    if (!test_constant_feature())
    {
        return 1;
    }

    printf("ALL NORMALIZATION TESTS PASSED!\n");
    printf("All allocated resources released.\n");

    return 0;
}
