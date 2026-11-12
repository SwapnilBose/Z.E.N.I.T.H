
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "za_network.h"

/*
 * IMPLEMENTATION 008
 *
 * Verify:
 *
 * Batch gradient = Average of individual gradients
 *
 * We compare parameter updates rather than accessing
 * gradients directly.
 */

/* Copy a network, including its exact weights and biases. */
static Network network_clone(const Network *source)
{
    Network copy = {0};

    if (source == NULL ||
        source->layers == NULL ||
        source->layer_count == 0)
    {
        return copy;
    }

    size_t *sizes = malloc(
        (source->layer_count + 1) * sizeof(size_t)
    );

    if (sizes == NULL)
        return copy;

    sizes[0] = source->layers[0].input_size;

    for (size_t i = 0; i < source->layer_count; i++)
        sizes[i + 1] = source->layers[i].output_size;

    copy = network_create(sizes, source->layer_count);

    free(sizes);

    if (copy.layers == NULL)
        return (Network){0};

    /* Replace random initialization with exact source values. */
    for (size_t l = 0; l < source->layer_count; l++)
    {
        DenseLayer *destination = &copy.layers[l];
        const DenseLayer *original = &source->layers[l];

        size_t weight_count =
            original->weights.rows * original->weights.cols;

        size_t bias_count =
            original->biases.rows * original->biases.cols;

        for (size_t i = 0; i < weight_count; i++)
        {
            destination->weights.data[i] =
                original->weights.data[i];
        }

        for (size_t i = 0; i < bias_count; i++)
        {
            destination->biases.data[i] =
                original->biases.data[i];
        }
    }

    return copy;
}


/* Compare batch parameters with averaged individual parameters. */
static float compare_parameter_matrix(
    const Matrix *batch,
    const float *individuals,
    size_t individual_count,
    size_t matrix_size)
{
    float max_difference = 0.0f;

    for (size_t p = 0; p < matrix_size; p++)
    {
        float average = 0.0f;

        for (size_t n = 0; n < individual_count; n++)
        {
            average += individuals[n * matrix_size + p];
        }

        average /= (float)individual_count;

        float difference = fabsf(batch->data[p] - average);

        if (difference > max_difference)
            max_difference = difference;
    }

    return max_difference;
}


int main(void)
{
    const size_t batch_size = 5;
    const size_t layer_sizes[] = {2, 3, 2};
    const size_t layer_count = 2;

    const float learning_rate = 0.1f;
    const float tolerance = 0.0001f;

    int passed = 1;

    Network original = {0};
    Network batch_network = {0};
    Network *individual_networks = NULL;

    Matrix inputs = {0};
    Matrix targets = {0};

    /*
     * STEP 1: Create deterministic starting weights.
     */
    srand(42);

    original = network_create(layer_sizes, layer_count);

    if (original.layers == NULL)
    {
        printf("ERROR: Could not create original network.\n");
        return 1;
    }

    /*
     * STEP 2: Clone the original network for batch training.
     */
    batch_network = network_clone(&original);

    if (batch_network.layers == NULL)
    {
        printf("ERROR: Could not clone batch network.\n");
        passed = 0;
        goto cleanup;
    }

    /*
     * STEP 3: Allocate individual networks.
     */
    individual_networks = calloc(
        batch_size,
        sizeof(Network)
    );

    if (individual_networks == NULL)
    {
        printf("ERROR: Could not allocate individual networks.\n");
        passed = 0;
        goto cleanup;
    }

    for (size_t i = 0; i < batch_size; i++)
    {
        individual_networks[i] = network_clone(&original);

        if (individual_networks[i].layers == NULL)
        {
            printf("ERROR: Could not clone network %zu.\n", i);
            passed = 0;
            goto cleanup;
        }
    }

    /*
     * STEP 4: Create five input-target examples.
     *
     * Each row is one training example.
     */
    inputs = matrix_create(batch_size, 2);
    targets = matrix_create(batch_size, 2);

    if (inputs.data == NULL || targets.data == NULL)
    {
        printf("ERROR: Could not allocate datasets.\n");
        passed = 0;
        goto cleanup;
    }

    const float input_values[] = {
         0.2f,  0.4f,
        -0.3f,  0.8f,
         0.7f, -0.2f,
         0.5f,  0.9f,
        -0.6f, -0.4f
    };

    const float target_values[] = {
         0.5f,  0.3f,
        -0.2f,  0.7f,
         0.8f, -0.1f,
         0.4f,  0.9f,
        -0.5f, -0.3f
    };

    for (size_t i = 0; i < batch_size * 2; i++)
    {
        inputs.data[i] = input_values[i];
        targets.data[i] = target_values[i];
    }

    /*
     * STEP 5: Train the batch network.
     *
     * All five examples are processed together.
     */
    float batch_loss = network_train_step(
        &batch_network,
        &inputs,
        &targets,
        learning_rate
    );

    if (!isfinite(batch_loss))
    {
        printf("ERROR: Batch training failed.\n");
        passed = 0;
        goto cleanup;
    }

    /*
     * STEP 6: Train each individual network.
     *
     * Every network starts from identical original weights.
     * Each network receives only ONE training example.
     */
    float individual_losses[5];

    for (size_t i = 0; i < batch_size; i++)
    {
        Matrix single_input = matrix_create(1, 2);
        Matrix single_target = matrix_create(1, 2);

        if (single_input.data == NULL ||
            single_target.data == NULL)
        {
            matrix_free(&single_input);
            matrix_free(&single_target);

            printf("ERROR: Could not create individual example.\n");
            passed = 0;
            goto cleanup;
        }

        for (size_t j = 0; j < 2; j++)
        {
            single_input.data[j] = inputs.data[i * 2 + j];
            single_target.data[j] = targets.data[i * 2 + j];
        }

        individual_losses[i] = network_train_step(
            &individual_networks[i],
            &single_input,
            &single_target,
            learning_rate
        );

        matrix_free(&single_input);
        matrix_free(&single_target);

        if (!isfinite(individual_losses[i]))
        {
            printf("ERROR: Individual training failed at %zu.\n", i);
            passed = 0;
            goto cleanup;
        }
    }

    /*
     * STEP 7: Compare batch parameters with the average
     * of all individually updated parameters.
     *
     * Since every network started from the same parameters
     * and used the same learning rate:
     *
     * Average(parameter_after_i)
     * =
     * parameter_before - learning_rate * average(gradient_i)
     *
     * This should equal the batch update.
     */
    float overall_max_difference = 0.0f;

    printf("\nGradient Equivalence Results\n");
    printf("----------------------------\n");

    printf("Batch loss: %.8f\n", batch_loss);

    for (size_t l = 0; l < layer_count; l++)
    {
        DenseLayer *batch_layer = &batch_network.layers[l];

        size_t weight_count =
            batch_layer->weights.rows *
            batch_layer->weights.cols;

        size_t bias_count =
            batch_layer->biases.rows *
            batch_layer->biases.cols;

        /*
         * Store all individual weights contiguously:
         * [network 0 weights][network 1 weights]...
         */
        float *individual_weights = malloc(
            batch_size * weight_count * sizeof(float)
        );

        float *individual_biases = malloc(
            batch_size * bias_count * sizeof(float)
        );

        if (individual_weights == NULL ||
            individual_biases == NULL)
        {
            free(individual_weights);
            free(individual_biases);

            printf("ERROR: Gradient comparison allocation failed.\n");
            passed = 0;
            goto cleanup;
        }

        for (size_t n = 0; n < batch_size; n++)
        {
            DenseLayer *layer = &individual_networks[n].layers[l];

            for (size_t p = 0; p < weight_count; p++)
            {
                individual_weights[n * weight_count + p] =
                    layer->weights.data[p];
            }

            for (size_t p = 0; p < bias_count; p++)
            {
                individual_biases[n * bias_count + p] =
                    layer->biases.data[p];
            }
        }

        float weight_difference = compare_parameter_matrix(
            &batch_layer->weights,
            individual_weights,
            batch_size,
            weight_count
        );

        float bias_difference = compare_parameter_matrix(
            &batch_layer->biases,
            individual_biases,
            batch_size,
            bias_count
        );

        printf("\nLayer %zu:\n", l + 1);
        printf("  Weight difference: %.10f\n", weight_difference);
        printf("  Bias difference:   %.10f\n", bias_difference);

        if (weight_difference > overall_max_difference)
            overall_max_difference = weight_difference;

        if (bias_difference > overall_max_difference)
            overall_max_difference = bias_difference;

        free(individual_weights);
        free(individual_biases);
    }

    /*
     * STEP 8: Final verdict.
     */
    printf("\nMaximum parameter difference: %.10f\n",
           overall_max_difference);

    if (overall_max_difference <= tolerance)
    {
        printf("\nGRADIENT EQUIVALENCE TEST PASSED!\n");
    }
    else
    {
        printf("\nGRADIENT EQUIVALENCE TEST FAILED!\n");
        passed = 0;
    }

cleanup:

    matrix_free(&inputs);
    matrix_free(&targets);

    network_free(&original);
    network_free(&batch_network);

    if (individual_networks != NULL)
    {
        for (size_t i = 0; i < batch_size; i++)
            network_free(&individual_networks[i]);

        free(individual_networks);
    }

    printf("\nAll allocated resources released.\n");

    return passed ? 0 : 1;
}
