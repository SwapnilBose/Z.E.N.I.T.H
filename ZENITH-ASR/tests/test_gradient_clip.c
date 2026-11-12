
#include "za_gradient.h"

#include <stdio.h>
#include <math.h>

#define TOLERANCE 0.0001f

static int nearly_equal(float a, float b)
{
    return fabsf(a - b) <= TOLERANCE;
}

/*
 * TEST 1:
 * Calculate global L2 norm.
 */
static int test_global_norm(void)
{
    printf("\nTEST 1: Global gradient norm\n");

    float data[] = {3.0f, 4.0f};

    Matrix gradient = {1, 2, data};

    double norm = gradient_global_norm(&gradient);

    printf("Calculated norm: %.6f\n", norm);

    int passed = fabs(norm - 5.0) < TOLERANCE;

    printf("%s\n", passed ? "PASSED" : "FAILED");

    return passed;
}

/*
 * TEST 2:
 * Clip a gradient whose norm exceeds max_norm.
 */
static int test_clipping(void)
{
    printf("\nTEST 2: Global norm clipping\n");

    float data[] = {3.0f, 4.0f};

    Matrix gradient = {1, 2, data};

    int success = gradient_clip_global_norm(&gradient, 2.0f);

    if (!success)
        return 0;

    printf("Clipped gradient: %.6f, %.6f\n",
           gradient.data[0], gradient.data[1]);

    double norm = gradient_global_norm(&gradient);

    printf("New norm: %.6f\n", norm);

    int passed =
        nearly_equal(gradient.data[0], 1.2f) &&
        nearly_equal(gradient.data[1], 1.6f) &&
        fabs(norm - 2.0) < TOLERANCE;

    printf("%s\n", passed ? "PASSED" : "FAILED");

    return passed;
}

/*
 * TEST 3:
 * A small gradient should remain unchanged.
 */
static int test_no_clipping_needed(void)
{
    printf("\nTEST 3: No clipping required\n");

    float data[] = {0.3f, 0.4f};

    Matrix gradient = {1, 2, data};

    int success = gradient_clip_global_norm(&gradient, 1.0f);

    int passed =
        success &&
        nearly_equal(gradient.data[0], 0.3f) &&
        nearly_equal(gradient.data[1], 0.4f);

    printf("Gradient: %.6f, %.6f\n",
           gradient.data[0], gradient.data[1]);

    printf("%s\n", passed ? "PASSED" : "FAILED");

    return passed;
}

/*
 * TEST 4:
 * Invalid clipping threshold must be rejected.
 */
static int test_invalid_threshold(void)
{
    printf("\nTEST 4: Invalid threshold\n");

    float data[] = {3.0f, 4.0f};

    Matrix gradient = {1, 2, data};

    int result_zero =
        gradient_clip_global_norm(&gradient, 0.0f);

    int result_negative =
        gradient_clip_global_norm(&gradient, -1.0f);

    int result_nan =
        gradient_clip_global_norm(&gradient, NAN);

    int passed =
        !result_zero &&
        !result_negative &&
        !result_nan &&
        nearly_equal(gradient.data[0], 3.0f) &&
        nearly_equal(gradient.data[1], 4.0f);

    printf("%s\n", passed ? "PASSED" : "FAILED");

    return passed;
}

/*
 * TEST 5:
 * Non-finite gradients must be rejected.
 */
static int test_non_finite_gradient(void)
{
    printf("\nTEST 5: Non-finite gradient rejection\n");

    float data[] = {3.0f, INFINITY};

    Matrix gradient = {1, 2, data};

    double norm = gradient_global_norm(&gradient);

    int result = gradient_clip_global_norm(&gradient, 2.0f);

    int passed = isnan(norm) && !result;

    printf("%s\n", passed ? "PASSED" : "FAILED");

    return passed;
}

/*
 * TEST 6:
 * Clipping must preserve gradient direction.
 *
 * Ratios between components must remain unchanged.
 */
static int test_direction_preservation(void)
{
    printf("\nTEST 6: Direction preservation\n");

    float data[] = {6.0f, 8.0f};

    Matrix gradient = {1, 2, data};

    int success = gradient_clip_global_norm(&gradient, 5.0f);

    if (!success)
        return 0;

    /*
     * Original ratio = 6 / 8 = 0.75
     * New ratio must also equal 0.75.
     */
    double original_ratio = 6.0 / 8.0;

    double clipped_ratio =
        (double)gradient.data[0] / gradient.data[1];

    int passed =
        fabs(original_ratio - clipped_ratio) < TOLERANCE;

    printf("Clipped gradient: %.6f, %.6f\n",
           gradient.data[0], gradient.data[1]);

    printf("%s\n", passed ? "PASSED" : "FAILED");

    return passed;
}

int main(void)
{
    int passed = 0;
    int total = 6;

    passed += test_global_norm();
    passed += test_clipping();
    passed += test_no_clipping_needed();
    passed += test_invalid_threshold();
    passed += test_non_finite_gradient();
    passed += test_direction_preservation();

    printf("\n============================\n");
    printf("RESULT: %d/%d tests passed\n", passed, total);
    printf("============================\n");

    return passed == total ? 0 : 1;
}
