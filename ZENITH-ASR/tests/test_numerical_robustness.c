
#include "za_matrix.h"
#include "za_softmax.h"
#include "za_loss.h"
#include "za_gradient.h"

#include <stdio.h>
#include <math.h>

#define TOLERANCE 0.0002

/*
 * Check whether a value is approximately equal
 * to an expected value.
 */
static int nearly_equal(double a, double b, double tolerance)
{
    return fabs(a - b) <= tolerance;
}


/*
 * TEST 1:
 * Softmax with extremely large positive logits.
 *
 * Maximum subtraction should prevent overflow.
 */
static int test_extreme_positive_logits(void)
{
    printf("\nTEST 1: Extreme positive logits\n");

    float data[] = {
        10000.0f, 9999.0f, 9998.0f
    };

    Matrix logits = {1, 3, data};

    Matrix probabilities = softmax_forward(&logits);

    if (!probabilities.data)
        return 0;

    double sum = 0.0;
    int passed = 1;

    for (size_t i = 0; i < 3; i++)
    {
        if (!isfinite(probabilities.data[i]))
            passed = 0;

        sum += probabilities.data[i];
    }

    if (!nearly_equal(sum, 1.0, TOLERANCE))
        passed = 0;

    printf("Probabilities: %.6f, %.6f, %.6f\n",
           probabilities.data[0],
           probabilities.data[1],
           probabilities.data[2]);

    printf("Probability sum: %.6f\n", sum);

    matrix_free(&probabilities);

    printf("%s\n", passed ? "PASSED" : "FAILED");

    return passed;
}


/*
 * TEST 2:
 * Softmax with extremely negative logits.
 *
 * The output should remain finite and normalized.
 */
static int test_extreme_negative_logits(void)
{
    printf("\nTEST 2: Extreme negative logits\n");

    float data[] = {
        -10000.0f, -10001.0f, -10002.0f
    };

    Matrix logits = {1, 3, data};

    Matrix probabilities = softmax_forward(&logits);

    if (!probabilities.data)
        return 0;

    double sum = 0.0;
    int passed = 1;

    for (size_t i = 0; i < 3; i++)
    {
        if (!isfinite(probabilities.data[i]))
            passed = 0;

        sum += probabilities.data[i];
    }

    if (!nearly_equal(sum, 1.0, TOLERANCE))
        passed = 0;

    printf("Probability sum: %.6f\n", sum);

    matrix_free(&probabilities);

    printf("%s\n", passed ? "PASSED" : "FAILED");

    return passed;
}


/*
 * TEST 3:
 * Cross-entropy with zero probability.
 *
 * log(0) is undefined.
 * Our loss implementation clamps small probabilities.
 */
static int test_near_zero_probability(void)
{
    printf("\nTEST 3: Near-zero probability protection\n");

    float p[] = {
        0.0f, 1.0f, 0.0f
    };

    float y[] = {
        1.0f, 0.0f, 0.0f
    };

    Matrix probabilities = {1, 3, p};
    Matrix targets = {1, 3, y};

    float loss = cross_entropy_loss(&probabilities, &targets);

    printf("Loss: %.6f\n", loss);

    int passed =
        isfinite(loss) &&
        nearly_equal(loss, -log(1.0e-7), 0.001);

    printf("%s\n", passed ? "PASSED" : "FAILED");

    return passed;
}


/*
 * TEST 4:
 * Invalid dimensions must be rejected.
 */
static int test_invalid_dimensions(void)
{
    printf("\nTEST 4: Invalid dimension handling\n");

    float p[] = {0.5f, 0.5f};
    float y[] = {1.0f, 0.0f, 0.0f};

    Matrix probabilities = {1, 2, p};
    Matrix targets = {1, 3, y};

    float loss = cross_entropy_loss(&probabilities, &targets);

    Matrix gradient =
        softmax_cross_entropy_gradient(&probabilities, &targets);

    int passed =
        isnan(loss) &&
        gradient.data == NULL;

    matrix_free(&gradient);

    printf("%s\n", passed ? "PASSED" : "FAILED");

    return passed;
}


/*
 * TEST 5:
 * NaN and infinity must be rejected.
 */
static int test_non_finite_values(void)
{
    printf("\nTEST 5: NaN and infinity rejection\n");

    float nan_data[] = {
        1.0f, NAN, 2.0f
    };

    float infinity_data[] = {
        1.0f, INFINITY, 2.0f
    };

    Matrix nan_logits = {1, 3, nan_data};
    Matrix infinity_logits = {1, 3, infinity_data};

    Matrix nan_output = softmax_forward(&nan_logits);
    Matrix infinity_output = softmax_forward(&infinity_logits);

    int passed =
        nan_output.data == NULL &&
        infinity_output.data == NULL;

    matrix_free(&nan_output);
    matrix_free(&infinity_output);

    printf("%s\n", passed ? "PASSED" : "FAILED");

    return passed;
}


/*
 * TEST 6:
 * Gradient clipping with very large finite values.
 *
 * Double precision norm accumulation should handle
 * these values without overflowing.
 */
static int test_large_gradient_clipping(void)
{
    printf("\nTEST 6: Large gradient clipping\n");

    float data[] = {
        3.0e30f, 4.0e30f
    };

    Matrix gradient = {1, 2, data};

    double original_norm = gradient_global_norm(&gradient);

    int success = gradient_clip_global_norm(&gradient, 1.0f);

    double clipped_norm = gradient_global_norm(&gradient);

    int passed =
        success &&
        isfinite(original_norm) &&
        nearly_equal(clipped_norm, 1.0, TOLERANCE) &&
        nearly_equal(gradient.data[0], 0.6, TOLERANCE) &&
        nearly_equal(gradient.data[1], 0.8, TOLERANCE);

    printf("Original norm: %.6e\n", original_norm);
    printf("Clipped norm: %.6f\n", clipped_norm);

    printf("Clipped values: %.6f, %.6f\n",
           gradient.data[0],
           gradient.data[1]);

    printf("%s\n", passed ? "PASSED" : "FAILED");

    return passed;
}


/*
 * TEST 7:
 * Integration consistency.
 *
 * Test the complete sequence:
 *
 *     logits -> softmax -> cross-entropy
 *                         -> gradient
 *
 * Check that the resulting values are finite.
 * Also verify that each sample's gradient sums to zero.
 */
static int test_full_classification_pipeline(void)
{
    printf("\nTEST 7: Full classification pipeline\n");

    float logits_data[] = {
         2.0f, 1.0f, 0.1f,
         0.5f, 2.0f, 1.0f
    };

    float targets_data[] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    };

    Matrix logits = {2, 3, logits_data};
    Matrix targets = {2, 3, targets_data};

    /* Step 1: Softmax. */
    Matrix probabilities = softmax_forward(&logits);

    if (!probabilities.data)
        return 0;

    /* Step 2: Cross-entropy. */
    float loss = cross_entropy_loss(&probabilities, &targets);

    /* Step 3: Gradient. */
    Matrix gradient =
        softmax_cross_entropy_gradient(&probabilities, &targets);

    if (!gradient.data)
    {
        matrix_free(&probabilities);
        return 0;
    }

    int passed = isfinite(loss);

    /* Verify every gradient is finite. */
    for (size_t i = 0; i < 6; i++)
    {
        if (!isfinite(gradient.data[i]))
            passed = 0;
    }

    /*
     * Each sample's gradient must sum to zero
     * because the softmax probabilities and
     * target distribution each sum to one.
     */
    for (size_t row = 0; row < 2; row++)
    {
        double row_sum = 0.0;

        for (size_t col = 0; col < 3; col++)
        {
            row_sum += gradient.data[row * 3 + col];
        }

        if (fabs(row_sum) > TOLERANCE)
            passed = 0;
    }

    printf("Cross-entropy loss: %.6f\n", loss);

    printf("Pipeline gradient:\n");

    for (size_t i = 0; i < 6; i++)
    {
        printf("%.6f ", gradient.data[i]);

        if ((i + 1) % 3 == 0)
            printf("\n");
    }

    matrix_free(&probabilities);
    matrix_free(&gradient);

    printf("%s\n", passed ? "PASSED" : "FAILED");

    return passed;
}


/*
 * Main test runner.
 */
int main(void)
{
    int passed = 0;
    int total = 7;

    passed += test_extreme_positive_logits();
    passed += test_extreme_negative_logits();
    passed += test_near_zero_probability();
    passed += test_invalid_dimensions();
    passed += test_non_finite_values();
    passed += test_large_gradient_clipping();
    passed += test_full_classification_pipeline();

    printf("\n================================\n");
    printf("NUMERICAL ROBUSTNESS RESULTS\n");
    printf("================================\n");

    printf("Passed: %d/%d\n", passed, total);

    printf("================================\n");

    return passed == total ? 0 : 1;
}
