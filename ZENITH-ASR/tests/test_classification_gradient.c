
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "za_network.h"
#include "za_softmax.h"
#include "za_loss.h"

/*
 * ZENITH-ASR
 * Implementation 012.3
 *
 * Classification gradient checking.
 *
 * Compares:
 *   1. Analytical gradients from backpropagation.
 *   2. Numerical gradients from central finite differences.
 *
 * Network: 2 -> 4 -> 3
 *
 * Every weight and bias is checked.
 */

#define SAMPLE_COUNT 3
#define INPUT_SIZE 2
#define CLASS_COUNT 3

#define LEARNING_RATE 0.001f
#define CLIP_THRESHOLD 1000000.0f

#define EPSILON 0.01f

#define ABS_TOLERANCE 0.002f
#define REL_TOLERANCE 0.02f

typedef struct
{
    size_t checked;
    size_t failed;
    double max_absolute_error;
    double max_relative_error;
} GradientCheckStats;


/*
 * Calculate classification loss without changing the network.
 */
static float evaluate_loss(
    const Network *network,
    const Matrix *inputs,
    const Matrix *targets
)
{
    Matrix logits = network_forward(network, inputs);

    if (logits.data == NULL)
        return NAN;

    Matrix probabilities = softmax_forward(&logits);

    matrix_free(&logits);

    if (probabilities.data == NULL)
        return NAN;

    float loss = cross_entropy_loss(
        &probabilities,
        targets
    );

    matrix_free(&probabilities);

    return loss;
}


/*
 * Save a copy of every layer's weights and biases.
 */
static int save_parameters(
    const Network *network,
    Matrix *saved_weights,
    Matrix *saved_biases
)
{
    for (size_t l = 0; l < network->layer_count; l++)
    {
        const DenseLayer *layer = &network->layers[l];

        saved_weights[l] = matrix_create(
            layer->weights.rows,
            layer->weights.cols
        );

        saved_biases[l] = matrix_create(
            layer->biases.rows,
            layer->biases.cols
        );

        if (saved_weights[l].data == NULL ||
            saved_biases[l].data == NULL)
        {
            return 0;
        }

        size_t weight_count =
            layer->weights.rows * layer->weights.cols;

        size_t bias_count =
            layer->biases.rows * layer->biases.cols;

        memcpy(
            saved_weights[l].data,
            layer->weights.data,
            weight_count * sizeof(float)
        );

        memcpy(
            saved_biases[l].data,
            layer->biases.data,
            bias_count * sizeof(float)
        );
    }

    return 1;
}


/*
 * Restore all network parameters from saved copies.
 */
static void restore_parameters(
    Network *network,
    const Matrix *saved_weights,
    const Matrix *saved_biases
)
{
    for (size_t l = 0; l < network->layer_count; l++)
    {
        DenseLayer *layer = &network->layers[l];

        size_t weight_count =
            layer->weights.rows * layer->weights.cols;

        size_t bias_count =
            layer->biases.rows * layer->biases.cols;

        memcpy(
            layer->weights.data,
            saved_weights[l].data,
            weight_count * sizeof(float)
        );

        memcpy(
            layer->biases.data,
            saved_biases[l].data,
            bias_count * sizeof(float)
        );
    }
}


/*
 * Recover analytical gradients from one gradient-descent step.
 *
 * Update rule:
 *
 *     new_parameter = old_parameter - learning_rate * gradient
 *
 * Therefore:
 *
 *     gradient = (old_parameter - new_parameter)
 *                / learning_rate
 */
static void recover_gradient(
    const Matrix *original,
    const Matrix *updated,
    Matrix *gradient,
    float learning_rate
)
{
    size_t count = original->rows * original->cols;

    for (size_t i = 0; i < count; i++)
    {
        gradient->data[i] =
            (original->data[i] - updated->data[i])
            / learning_rate;
    }
}


/*
 * Compare analytical and numerical gradients for one matrix.
 *
 * The parameter matrix is temporarily modified.
 * Every element is restored after checking.
 */
static void check_matrix(
    Network *network,
    Matrix *parameters,
    const Matrix *analytical,
    const Matrix *inputs,
    const Matrix *targets,
    const char *parameter_name,
    size_t layer_index,
    GradientCheckStats *stats
)
{
    size_t count = parameters->rows * parameters->cols;

    for (size_t i = 0; i < count; i++)
    {
        float original_value = parameters->data[i];

        /*
         * Evaluate L(w + epsilon).
         */
        parameters->data[i] =
            original_value + EPSILON;

        float loss_plus = evaluate_loss(
            network,
            inputs,
            targets
        );

        /*
         * Evaluate L(w - epsilon).
         */
        parameters->data[i] =
            original_value - EPSILON;

        float loss_minus = evaluate_loss(
            network,
            inputs,
            targets
        );

        /*
         * Restore the original parameter.
         */
        parameters->data[i] = original_value;

        if (!isfinite(loss_plus) || !isfinite(loss_minus))
        {
            printf(
                "ERROR: Invalid numerical loss at layer %zu, %s[%zu].\n",
                layer_index,
                parameter_name,
                i
            );

            stats->failed++;
            stats->checked++;

            continue;
        }

        /*
         * Central finite difference.
         */
        double numerical =
            ((double)loss_plus - (double)loss_minus)
            / (2.0 * (double)EPSILON);

        double expected = analytical->data[i];

        double absolute_error = fabs(
            numerical - expected
        );

        double denominator = fmax(
            fabs(numerical),
            fabs(expected)
        );

        double relative_error =
            absolute_error / fmax(denominator, 1e-8);

        if (absolute_error > stats->max_absolute_error)
            stats->max_absolute_error = absolute_error;

        if (relative_error > stats->max_relative_error)
            stats->max_relative_error = relative_error;

        /*
         * Combined absolute and relative tolerance.
         */
        double tolerance =
            ABS_TOLERANCE +
            REL_TOLERANCE * denominator;

        if (absolute_error > tolerance)
        {
            stats->failed++;

            printf(
                "FAIL | Layer %zu | %s[%zu]\n",
                layer_index,
                parameter_name,
                i
            );

            printf(
                "     Analytical: %.8f\n",
                expected
            );

            printf(
                "     Numerical:  %.8f\n",
                numerical
            );

            printf(
                "     Difference: %.8f\n",
                absolute_error
            );
        }

        stats->checked++;
    }
}


int main(void)
{
    printf("========================================\n");
    printf(" ZENITH CLASSIFICATION GRADIENT CHECK\n");
    printf(" Implementation 012.3\n");
    printf("========================================\n\n");

    srand(42);

    /*
     * STEP 1: CREATE A SMALL DATASET.
     */
    Matrix inputs = matrix_create(
        SAMPLE_COUNT,
        INPUT_SIZE
    );

    Matrix targets = matrix_create(
        SAMPLE_COUNT,
        CLASS_COUNT
    );

    if (inputs.data == NULL || targets.data == NULL)
    {
        printf("ERROR: Dataset allocation failed.\n");

        matrix_free(&inputs);
        matrix_free(&targets);

        return 1;
    }

    /*
     * Input samples.
     */
    float input_values[] =
    {
        -0.8f,  0.3f,
         0.2f,  0.9f,
         0.7f, -0.5f
    };

    /*
     * One-hot targets.
     */
    float target_values[] =
    {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };

    memcpy(
        inputs.data,
        input_values,
        sizeof(input_values)
    );

    memcpy(
        targets.data,
        target_values,
        sizeof(target_values)
    );

    /*
     * STEP 2: CREATE NETWORK.
     */
    size_t architecture[] = {2, 4, 3};

    Network network = network_create(
        architecture,
        2
    );

    if (network.layers == NULL)
    {
        printf("ERROR: Network creation failed.\n");

        matrix_free(&inputs);
        matrix_free(&targets);

        return 1;
    }

    printf("Network: 2 -> 4 -> 3\n");
    printf("Samples: %d\n", SAMPLE_COUNT);
    printf("Learning rate: %.6f\n", LEARNING_RATE);
    printf("Finite difference epsilon: %.6f\n\n", EPSILON);

    /*
     * STEP 3: SAVE ORIGINAL PARAMETERS.
     */
    Matrix *saved_weights = calloc(
        network.layer_count,
        sizeof(Matrix)
    );

    Matrix *saved_biases = calloc(
        network.layer_count,
        sizeof(Matrix)
    );

    Matrix *analytical_weights = calloc(
        network.layer_count,
        sizeof(Matrix)
    );

    Matrix *analytical_biases = calloc(
        network.layer_count,
        sizeof(Matrix)
    );

    if (saved_weights == NULL ||
        saved_biases == NULL ||
        analytical_weights == NULL ||
        analytical_biases == NULL)
    {
        printf("ERROR: Snapshot allocation failed.\n");

        free(saved_weights);
        free(saved_biases);
        free(analytical_weights);
        free(analytical_biases);

        network_free(&network);
        matrix_free(&inputs);
        matrix_free(&targets);

        return 1;
    }

    if (!save_parameters(
            &network,
            saved_weights,
            saved_biases))
    {
        printf("ERROR: Could not save network parameters.\n");

        goto cleanup_failure;
    }

    /*
     * STEP 4: ALLOCATE ANALYTICAL GRADIENT MATRICES.
     */
    for (size_t l = 0; l < network.layer_count; l++)
    {
        analytical_weights[l] = matrix_create(
            network.layers[l].weights.rows,
            network.layers[l].weights.cols
        );

        analytical_biases[l] = matrix_create(
            network.layers[l].biases.rows,
            network.layers[l].biases.cols
        );

        if (analytical_weights[l].data == NULL ||
            analytical_biases[l].data == NULL)
        {
            printf("ERROR: Gradient allocation failed.\n");

            goto cleanup_failure;
        }
    }

    /*
     * STEP 5: PERFORM ONE TRAINING STEP.
     *
     * The very large clipping threshold makes clipping
     * inactive for this small test network.
     */
    float training_loss = network_train_classification_step(
        &network,
        &inputs,
        &targets,
        LEARNING_RATE,
        CLIP_THRESHOLD
    );

    if (!isfinite(training_loss))
    {
        printf("ERROR: Classification training step failed.\n");

        goto cleanup_failure;
    }

    /*
     * STEP 6: RECOVER ANALYTICAL GRADIENTS.
     */
    for (size_t l = 0; l < network.layer_count; l++)
    {
        recover_gradient(
            &saved_weights[l],
            &network.layers[l].weights,
            &analytical_weights[l],
            LEARNING_RATE
        );

        recover_gradient(
            &saved_biases[l],
            &network.layers[l].biases,
            &analytical_biases[l],
            LEARNING_RATE
        );
    }

    /*
     * Restore the exact original network before numerical
     * gradient calculations.
     */
    restore_parameters(
        &network,
        saved_weights,
        saved_biases
    );

    printf("Analytical gradients recovered.\n");
    printf("Original network parameters restored.\n\n");

    /*
     * STEP 7: NUMERICAL GRADIENT CHECKING.
     */
    GradientCheckStats stats = {0};

    printf("Checking gradients...\n\n");

    for (size_t l = 0; l < network.layer_count; l++)
    {
        check_matrix(
            &network,
            &network.layers[l].weights,
            &analytical_weights[l],
            &inputs,
            &targets,
            "weights",
            l,
            &stats
        );

        check_matrix(
            &network,
            &network.layers[l].biases,
            &analytical_biases[l],
            &inputs,
            &targets,
            "biases",
            l,
            &stats
        );
    }

    /*
     * STEP 8: RESULTS.
     */
    printf("\n========================================\n");
    printf(" GRADIENT CHECK RESULTS\n");
    printf("========================================\n");

    printf("Parameters checked: %zu\n", stats.checked);
    printf("Failed checks:     %zu\n", stats.failed);

    printf(
        "Maximum absolute error: %.10f\n",
        stats.max_absolute_error
    );

    printf(
        "Maximum relative error: %.10f\n",
        stats.max_relative_error
    );

    int passed =
        stats.checked > 0 &&
        stats.failed == 0;

    if (passed)
        printf("\nCLASSIFICATION GRADIENT CHECK PASSED\n");
    else
        printf("\nCLASSIFICATION GRADIENT CHECK FAILED\n");

    printf("========================================\n");

    /*
     * STEP 9: CLEANUP.
     */
    for (size_t l = 0; l < network.layer_count; l++)
    {
        matrix_free(&saved_weights[l]);
        matrix_free(&saved_biases[l]);

        matrix_free(&analytical_weights[l]);
        matrix_free(&analytical_biases[l]);
    }

    free(saved_weights);
    free(saved_biases);
    free(analytical_weights);
    free(analytical_biases);

    network_free(&network);
    matrix_free(&inputs);
    matrix_free(&targets);

    return passed ? 0 : 1;


cleanup_failure:

    for (size_t l = 0; l < network.layer_count; l++)
    {
        matrix_free(&saved_weights[l]);
        matrix_free(&saved_biases[l]);

        matrix_free(&analytical_weights[l]);
        matrix_free(&analytical_biases[l]);
    }

    free(saved_weights);
    free(saved_biases);
    free(analytical_weights);
    free(analytical_biases);

    network_free(&network);
    matrix_free(&inputs);
    matrix_free(&targets);

    return 1;
}
