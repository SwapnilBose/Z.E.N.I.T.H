
#include "za_loss.h"
#include "za_softmax.h"

#include <stdio.h>
#include <math.h>

/* Compare two floating-point values. */
#define TOLERANCE 0.0002f

static int nearly_equal(float a, float b, float tolerance)
{
    return fabsf(a - b) <= tolerance;
}

/*
 * TEST 1:
 * Basic softmax + cross-entropy gradient.
 */
static int test_basic_gradient(void)
{
    printf("\nTEST 1: Basic gradient\n");

    float p[] = {0.8f, 0.1f, 0.1f};
    float y[] = {1.0f, 0.0f, 0.0f};

    Matrix probabilities = {1, 3, p};
    Matrix targets = {1, 3, y};

    Matrix gradient =
        softmax_cross_entropy_gradient(&probabilities, &targets);

    if (!gradient.data)
        return 0;

    float expected[] = {-0.2f, 0.1f, 0.1f};

    int passed = 1;

    for (size_t i = 0; i < 3; i++)
    {
        printf("Gradient[%zu] = %.6f\n", i, gradient.data[i]);

        if (!nearly_equal(gradient.data[i], expected[i], TOLERANCE))
            passed = 0;
    }

    matrix_free(&gradient);

    printf("%s\n", passed ? "PASSED" : "FAILED");

    return passed;
}

/*
 * TEST 2:
 * Batch gradient averaging.
 */
static int test_batch_gradient(void)
{
    printf("\nTEST 2: Batch gradient\n");

    float p[] = {
        0.8f, 0.1f, 0.1f,
        0.1f, 0.8f, 0.1f
    };

    float y[] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    };

    Matrix probabilities = {2, 3, p};
    Matrix targets = {2, 3, y};

    Matrix gradient =
        softmax_cross_entropy_gradient(&probabilities, &targets);

    if (!gradient.data)
        return 0;

    float expected[] = {
        -0.1f, 0.05f, 0.05f,
         0.05f, -0.1f, 0.05f
    };

    int passed = 1;

    for (size_t i = 0; i < 6; i++)
    {
        printf("Gradient[%zu] = %.6f\n", i, gradient.data[i]);

        if (!nearly_equal(gradient.data[i], expected[i], TOLERANCE))
            passed = 0;
    }

    matrix_free(&gradient);

    printf("%s\n", passed ? "PASSED" : "FAILED");

    return passed;
}

/*
 * TEST 3:
 * Numerical gradient checking.
 *
 * Compare:
 * Analytical gradient
 * against
 * Central finite-difference gradient.
 */
static int test_numerical_gradient(void)
{
    printf("\nTEST 3: Numerical gradient checking\n");

    float logits_data[] = {
         1.2f, 0.3f, -0.7f,
        -0.2f, 0.8f,  0.1f
    };

    float targets_data[] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    };

    Matrix logits = {2, 3, logits_data};
    Matrix targets = {2, 3, targets_data};

    Matrix probabilities = softmax_forward(&logits);

    if (!probabilities.data)
        return 0;

    Matrix analytical =
        softmax_cross_entropy_gradient(&probabilities, &targets);

    if (!analytical.data)
    {
        matrix_free(&probabilities);
        return 0;
    }

    const float h = 0.001f;

    int passed = 1;
    float max_difference = 0.0f;

    for (size_t i = 0; i < 6; i++)
    {
        float original = logits.data[i];

        /* Calculate loss at z + h. */
        logits.data[i] = original + h;

        Matrix p_plus = softmax_forward(&logits);

        if (!p_plus.data)
        {
            passed = 0;
            break;
        }

        float loss_plus = cross_entropy_loss(&p_plus, &targets);

        matrix_free(&p_plus);

        /* Calculate loss at z - h. */
        logits.data[i] = original - h;

        Matrix p_minus = softmax_forward(&logits);

        if (!p_minus.data)
        {
            logits.data[i] = original;
            passed = 0;
            break;
        }

        float loss_minus = cross_entropy_loss(&p_minus, &targets);

        matrix_free(&p_minus);

        /* Restore original logit. */
        logits.data[i] = original;

        /* Central finite-difference approximation. */
        float numerical =
            (loss_plus - loss_minus) / (2.0f * h);

        float difference =
            fabsf(numerical - analytical.data[i]);

        if (difference > max_difference)
            max_difference = difference;

        printf(
            "Parameter %zu: analytical = %.6f, numerical = %.6f\n",
            i,
            analytical.data[i],
            numerical
        );

        if (difference > TOLERANCE)
            passed = 0;
    }

    printf("Maximum difference: %.8f\n", max_difference);

    matrix_free(&analytical);
    matrix_free(&probabilities);

    printf("%s\n", passed ? "PASSED" : "FAILED");

    return passed;
}

int main(void)
{
    int passed = 0;
    int total = 3;

    passed += test_basic_gradient();
    passed += test_batch_gradient();
    passed += test_numerical_gradient();

    printf("\n============================\n");
    printf("RESULT: %d/%d tests passed\n", passed, total);
    printf("============================\n");

    return (passed == total) ? 0 : 1;
}
