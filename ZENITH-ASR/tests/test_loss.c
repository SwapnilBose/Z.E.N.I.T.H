
#include <stdio.h>
#include <math.h>

#include "za_matrix.h"
#include "za_loss.h"

/*
 * Compare two floating-point values within tolerance.
 */
static int nearly_equal(float a, float b, float tolerance)
{
    return isfinite(a) &&
           isfinite(b) &&
           fabsf(a - b) <= tolerance;
}


int main(void)
{
    int passed = 0;
    const int total = 5;

    Matrix probabilities = {0};
    Matrix targets = {0};

    float loss;


    /*
     * TEST 1:
     * Correct prediction.
     *
     * Prediction: [0.8, 0.1, 0.1]
     * Target:     [1.0, 0.0, 0.0]
     *
     * Expected loss:
     * -log(0.8) = 0.22314355
     */
    printf("\n[TEST 1] Correct prediction...\n");

    probabilities = matrix_create(1, 3);
    targets = matrix_create(1, 3);

    if (!probabilities.data || !targets.data)
        goto cleanup;

    probabilities.data[0] = 0.8f;
    probabilities.data[1] = 0.1f;
    probabilities.data[2] = 0.1f;

    targets.data[0] = 1.0f;
    targets.data[1] = 0.0f;
    targets.data[2] = 0.0f;

    loss = cross_entropy_loss(&probabilities, &targets);

    printf("Loss: %.8f\n", loss);

    if (nearly_equal(loss, 0.22314355f, 0.0001f))
    {
        printf("PASS\n");
        passed++;
    }
    else
    {
        printf("FAIL\n");
    }

    matrix_free(&probabilities);
    matrix_free(&targets);


    /*
     * TEST 2:
     * Incorrect prediction.
     *
     * Prediction: [0.1, 0.8, 0.1]
     * Target:     [1.0, 0.0, 0.0]
     *
     * Expected loss:
     * -log(0.1) = 2.30258509
     */
    printf("\n[TEST 2] Incorrect prediction...\n");

    probabilities = matrix_create(1, 3);
    targets = matrix_create(1, 3);

    if (!probabilities.data || !targets.data)
        goto cleanup;

    probabilities.data[0] = 0.1f;
    probabilities.data[1] = 0.8f;
    probabilities.data[2] = 0.1f;

    targets.data[0] = 1.0f;
    targets.data[1] = 0.0f;
    targets.data[2] = 0.0f;

    loss = cross_entropy_loss(&probabilities, &targets);

    printf("Loss: %.8f\n", loss);

    if (nearly_equal(loss, 2.30258509f, 0.0001f))
    {
        printf("PASS\n");
        passed++;
    }
    else
    {
        printf("FAIL\n");
    }

    matrix_free(&probabilities);
    matrix_free(&targets);


    /*
     * TEST 3:
     * Batch averaging.
     *
     * Sample 1 loss = -log(0.8)
     * Sample 2 loss = -log(0.1)
     *
     * Expected:
     *
     * (-log(0.8) - log(0.1)) / 2
     * = 1.26286432
     */
    printf("\n[TEST 3] Batch loss averaging...\n");

    probabilities = matrix_create(2, 3);
    targets = matrix_create(2, 3);

    if (!probabilities.data || !targets.data)
        goto cleanup;

    float batch_probabilities[] =
    {
        0.8f, 0.1f, 0.1f,
        0.1f, 0.8f, 0.1f
    };

    float batch_targets[] =
    {
        1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f
    };

    for (size_t i = 0; i < 6; i++)
    {
        probabilities.data[i] = batch_probabilities[i];
        targets.data[i] = batch_targets[i];
    }

    loss = cross_entropy_loss(&probabilities, &targets);

    printf("Mean loss: %.8f\n", loss);

    if (nearly_equal(loss, 1.26286432f, 0.0001f))
    {
        printf("PASS\n");
        passed++;
    }
    else
    {
        printf("FAIL\n");
    }

    matrix_free(&probabilities);
    matrix_free(&targets);


    /*
     * TEST 4:
     * Zero-probability protection.
     *
     * Prediction: [1.0, 0.0, 0.0]
     * Target:     [0.0, 1.0, 0.0]
     *
     * The correct class has probability zero.
     * The loss should remain finite because of epsilon.
     */
    printf("\n[TEST 4] Zero-probability protection...\n");

    probabilities = matrix_create(1, 3);
    targets = matrix_create(1, 3);

    if (!probabilities.data || !targets.data)
        goto cleanup;

    probabilities.data[0] = 1.0f;
    probabilities.data[1] = 0.0f;
    probabilities.data[2] = 0.0f;

    targets.data[0] = 0.0f;
    targets.data[1] = 1.0f;
    targets.data[2] = 0.0f;

    loss = cross_entropy_loss(&probabilities, &targets);

    printf("Loss: %.8f\n", loss);

    if (isfinite(loss) && loss > 10.0f)
    {
        printf("PASS\n");
        passed++;
    }
    else
    {
        printf("FAIL\n");
    }

    matrix_free(&probabilities);
    matrix_free(&targets);


    /*
     * TEST 5:
     * Invalid distribution detection.
     *
     * The target contains NaN.
     * The function should reject the input.
     */
    printf("\n[TEST 5] Invalid input detection...\n");

    probabilities = matrix_create(1, 3);
    targets = matrix_create(1, 3);

    if (!probabilities.data || !targets.data)
        goto cleanup;

    probabilities.data[0] = 0.8f;
    probabilities.data[1] = 0.1f;
    probabilities.data[2] = 0.1f;

    targets.data[0] = 1.0f;
    targets.data[1] = NAN;
    targets.data[2] = 0.0f;

    loss = cross_entropy_loss(&probabilities, &targets);

    if (isnan(loss))
    {
        printf("Invalid target correctly rejected.\n");
        printf("PASS\n");
        passed++;
    }
    else
    {
        printf("FAIL\n");
    }


cleanup:

    matrix_free(&probabilities);
    matrix_free(&targets);

    printf("\n================================\n");
    printf(" CROSS-ENTROPY TESTS: %d / %d PASSED\n",
           passed, total);
    printf("================================\n");

    return passed == total ? 0 : 1;
}
