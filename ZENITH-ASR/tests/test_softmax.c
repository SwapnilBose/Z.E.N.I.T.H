
#include <stdio.h>
#include <math.h>

#include "za_matrix.h"
#include "za_softmax.h"

/*
 * Check whether all probabilities in each row
 * sum to approximately 1.
 */
static int check_row_sums(const Matrix *matrix)
{
    if (!matrix || !matrix->data)
        return 0;

    for (size_t row = 0; row < matrix->rows; row++)
    {
        double sum = 0.0;

        for (size_t col = 0; col < matrix->cols; col++)
        {
            float value = matrix->data[row * matrix->cols + col];

            if (!isfinite(value) || value < 0.0f || value > 1.0f)
                return 0;

            sum += value;
        }

        if (fabs(sum - 1.0) > 0.00001)
            return 0;
    }

    return 1;
}


int main(void)
{
    int passed = 0;
    int total = 4;

    Matrix logits = {0};
    Matrix probabilities = {0};

    /*
     * TEST 1:
     * Normal logits.
     *
     * Expected approximate output:
     * [0.6652, 0.2447, 0.0900]
     */
    printf("\n[TEST 1] Normal softmax calculation...\n");

    logits = matrix_create(1, 3);

    if (!logits.data)
        goto cleanup;

    logits.data[0] = 2.0f;
    logits.data[1] = 1.0f;
    logits.data[2] = 0.0f;

    probabilities = softmax_forward(&logits);

    if (!probabilities.data)
        goto cleanup;

    printf("Output: ");

    for (size_t i = 0; i < probabilities.cols; i++)
        printf("%.6f ", probabilities.data[i]);

    printf("\n");

    if (check_row_sums(&probabilities) &&
        fabsf(probabilities.data[0] - 0.665241f) < 0.0001f &&
        fabsf(probabilities.data[1] - 0.244728f) < 0.0001f &&
        fabsf(probabilities.data[2] - 0.090031f) < 0.0001f)
    {
        printf("PASS\n");
        passed++;
    }
    else
    {
        printf("FAIL\n");
    }

    matrix_free(&probabilities);
    matrix_free(&logits);


    /*
     * TEST 2:
     * Extremely large logits.
     *
     * Direct exp(1000) would overflow.
     * Stable softmax should still work.
     */
    printf("\n[TEST 2] Large logits...\n");

    logits = matrix_create(1, 3);

    if (!logits.data)
        goto cleanup;

    logits.data[0] = 1000.0f;
    logits.data[1] = 1001.0f;
    logits.data[2] = 999.0f;

    probabilities = softmax_forward(&logits);

    if (probabilities.data &&
        check_row_sums(&probabilities))
    {
        printf("Output: ");

        for (size_t i = 0; i < probabilities.cols; i++)
            printf("%.6f ", probabilities.data[i]);

        printf("\nPASS\n");
        passed++;
    }
    else
    {
        printf("FAIL\n");
    }

    matrix_free(&probabilities);
    matrix_free(&logits);


    /*
     * TEST 3:
     * Batch processing.
     *
     * Every row must independently sum to 1.
     */
    printf("\n[TEST 3] Batch softmax...\n");

    logits = matrix_create(3, 3);

    if (!logits.data)
        goto cleanup;

    float batch_values[] =
    {
        2.0f, 1.0f, 0.0f,
        1.0f, 2.0f, 3.0f,
        5.0f, 5.0f, 5.0f
    };

    for (size_t i = 0; i < 9; i++)
        logits.data[i] = batch_values[i];

    probabilities = softmax_forward(&logits);

    if (probabilities.data &&
        check_row_sums(&probabilities))
    {
        printf("All three rows sum to approximately 1.\n");
        printf("PASS\n");
        passed++;
    }
    else
    {
        printf("FAIL\n");
    }

    matrix_free(&probabilities);
    matrix_free(&logits);


    /*
     * TEST 4:
     * Invalid input containing NaN.
     *
     * Softmax should reject it.
     */
    printf("\n[TEST 4] Invalid input detection...\n");

    logits = matrix_create(1, 3);

    if (!logits.data)
        goto cleanup;

    logits.data[0] = 1.0f;
    logits.data[1] = NAN;
    logits.data[2] = 3.0f;

    probabilities = softmax_forward(&logits);

    if (!probabilities.data)
    {
        printf("Invalid input correctly rejected.\n");
        printf("PASS\n");
        passed++;
    }
    else
    {
        printf("FAIL: Invalid input was accepted.\n");
    }

    matrix_free(&probabilities);
    matrix_free(&logits);


cleanup:

    matrix_free(&probabilities);
    matrix_free(&logits);

    printf("\n================================\n");
    printf(" SOFTMAX TESTS: %d / %d PASSED\n", passed, total);
    printf("================================\n");

    return passed == total ? 0 : 1;
}
