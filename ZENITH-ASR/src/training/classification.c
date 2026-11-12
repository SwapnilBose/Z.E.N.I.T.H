
#include "za_network.h"

#include "za_softmax.h"
#include "za_loss.h"
#include "za_gradient.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/*
 * Check whether a matrix has valid allocated storage.
 */
static int matrix_is_valid(const Matrix *m)
{
    return m != NULL &&
           m->data != NULL &&
           m->rows > 0 &&
           m->cols > 0;
}

/*
 * Validate the network's structure and dimensions.
 */
static int network_is_valid(const Network *network)
{
    size_t l;

    if (network == NULL ||
        network->layers == NULL ||
        network->layer_count == 0)
    {
        return 0;
    }

    for (l = 0; l < network->layer_count; l++)
    {
        const DenseLayer *layer = &network->layers[l];

        if (layer->input_size == 0 ||
            layer->output_size == 0 ||
            !matrix_is_valid(&layer->weights) ||
            !matrix_is_valid(&layer->biases))
        {
            return 0;
        }

        if (layer->weights.rows != layer->input_size ||
            layer->weights.cols != layer->output_size ||
            layer->biases.rows != 1 ||
            layer->biases.cols != layer->output_size)
        {
            return 0;
        }

        if (l > 0 &&
            network->layers[l - 1].output_size !=
            layer->input_size)
        {
            return 0;
        }
    }

    return 1;
}

/*
 * Release an array of matrices.
 */
static void free_matrix_array(Matrix *matrices, size_t count)
{
    size_t i;

    if (matrices == NULL)
        return;

    for (i = 0; i < count; i++)
        matrix_free(&matrices[i]);

    free(matrices);
}

/*
 * Verify that every matrix element is finite.
 */
static int matrix_values_are_finite(const Matrix *m)
{
    size_t count;

    if (!matrix_is_valid(m))
        return 0;

    if (m->rows > SIZE_MAX / m->cols)
        return 0;

    count = m->rows * m->cols;

    for (size_t i = 0; i < count; i++)
    {
        if (!isfinite(m->data[i]))
            return 0;
    }

    return 1;
}

/*
 * Calculate one classification training step.
 *
 * Architecture:
 *
 * Input
 *   |
 * Dense + ReLU
 *   |
 * Dense + ReLU
 *   |
 * Final Dense (linear logits)
 *   |
 * Softmax
 *   |
 * Cross-entropy
 *   |
 * Backpropagation
 *   |
 * Global gradient clipping
 *   |
 * Parameter update
 */
float network_train_classification_step(
    Network *network,
    const Matrix *input,
    const Matrix *one_hot_targets,
    float learning_rate,
    float max_gradient_norm)
{
    size_t layer_count;
    size_t batch_size;
    size_t output_size;

    size_t l, r, i, j, k;

    Matrix *activations = NULL;
    Matrix *preactivations = NULL;
    Matrix *deltas = NULL;

    Matrix *weight_grads = NULL;
    Matrix *bias_grads = NULL;

    Matrix probabilities = {0};

    double global_norm = 0.0;
    double clip_scale = 1.0;

    float loss = NAN;
    int success = 0;

    /*
     * PHASE 0: VALIDATE INPUTS
     */
    if (!network_is_valid(network) ||
        !matrix_is_valid(input) ||
        !matrix_is_valid(one_hot_targets) ||
        !isfinite(learning_rate) ||
        learning_rate <= 0.0f ||
        !isfinite(max_gradient_norm) ||
        max_gradient_norm <= 0.0f)
    {
        return NAN;
    }

    layer_count = network->layer_count;
    batch_size = input->rows;

    output_size =
        network->layers[layer_count - 1].output_size;

    if (input->cols != network->layers[0].input_size ||
        one_hot_targets->rows != batch_size ||
        one_hot_targets->cols != output_size)
    {
        return NAN;
    }

    if (layer_count == SIZE_MAX)
        return NAN;

    if (!matrix_values_are_finite(input))
        return NAN;

    /*
     * PHASE 1: ALLOCATE TEMPORARY ARRAYS
     */
    activations = calloc(layer_count + 1, sizeof(Matrix));
    preactivations = calloc(layer_count, sizeof(Matrix));
    deltas = calloc(layer_count, sizeof(Matrix));

    weight_grads = calloc(layer_count, sizeof(Matrix));
    bias_grads = calloc(layer_count, sizeof(Matrix));

    if (activations == NULL ||
        preactivations == NULL ||
        deltas == NULL ||
        weight_grads == NULL ||
        bias_grads == NULL)
    {
        goto cleanup;
    }

    /*
     * The input is borrowed.
     * This function must not free it.
     */
    activations[0] = *input;

    /*
     * PHASE 2: FORWARD PROPAGATION
     *
     * Hidden layers use ReLU.
     * Final layer produces raw logits.
     */
    for (l = 0; l < layer_count; l++)
    {
        const DenseLayer *layer = &network->layers[l];

        size_t in_size = layer->input_size;
        size_t out_size = layer->output_size;

        preactivations[l] =
            matrix_create(batch_size, out_size);

        activations[l + 1] =
            matrix_create(batch_size, out_size);

        if (!matrix_is_valid(&preactivations[l]) ||
            !matrix_is_valid(&activations[l + 1]))
        {
            goto cleanup;
        }

        for (r = 0; r < batch_size; r++)
        {
            for (j = 0; j < out_size; j++)
            {
                double sum = layer->biases.data[j];

                for (i = 0; i < in_size; i++)
                {
                    sum +=
                        (double)activations[l].data[
                            r * in_size + i
                        ] *
                        layer->weights.data[
                            i * out_size + j
                        ];
                }

                if (!isfinite(sum))
                    goto cleanup;

                float z = (float)sum;

                if (!isfinite(z))
                    goto cleanup;

                preactivations[l].data[
                    r * out_size + j
                ] = z;

                /*
                 * Apply ReLU only to hidden layers.
                 * The final layer remains linear.
                 */
                if (l + 1 < layer_count && z < 0.0f)
                    z = 0.0f;

                activations[l + 1].data[
                    r * out_size + j
                ] = z;
            }
        }
    }

    /*
     * PHASE 3: SOFTMAX
     *
     * Final activations are logits.
     */
    probabilities =
        softmax_forward(&activations[layer_count]);

    if (!matrix_is_valid(&probabilities))
        goto cleanup;

    /*
     * PHASE 4: CROSS-ENTROPY LOSS
     */
    loss = cross_entropy_loss(
        &probabilities,
        one_hot_targets
    );

    if (!isfinite(loss))
        goto cleanup;

    /*
     * PHASE 5: OUTPUT DELTA
     *
     * The combined softmax + cross-entropy derivative is:
     *
     * delta = (probability - target) / batch_size
     *
     * This is already averaged over the batch.
     */
    deltas[layer_count - 1] =
        softmax_cross_entropy_gradient(
            &probabilities,
            one_hot_targets
        );

    if (!matrix_is_valid(&deltas[layer_count - 1]))
        goto cleanup;

    /*
     * PHASE 6: BACKPROPAGATION THROUGH HIDDEN LAYERS
     */
    for (l = layer_count - 1; l-- > 0;)
    {
        size_t current_size =
            network->layers[l].output_size;

        size_t next_size =
            network->layers[l + 1].output_size;

        deltas[l] =
            matrix_create(batch_size, current_size);

        if (!matrix_is_valid(&deltas[l]))
            goto cleanup;

        for (r = 0; r < batch_size; r++)
        {
            for (i = 0; i < current_size; i++)
            {
                double sum = 0.0;

                /*
                 * Propagate the next layer's error backward.
                 */
                for (k = 0; k < next_size; k++)
                {
                    sum +=
                        (double)deltas[l + 1].data[
                            r * next_size + k
                        ] *
                        network->layers[l + 1].weights.data[
                            i * next_size + k
                        ];
                }

                /*
                 * Derivative of ReLU:
                 *
                 * z > 0  => 1
                 * z <= 0 => 0
                 */
                if (preactivations[l].data[
                        r * current_size + i
                    ] <= 0.0f)
                {
                    sum = 0.0;
                }

                float value = (float)sum;

                if (!isfinite(value))
                    goto cleanup;

                deltas[l].data[
                    r * current_size + i
                ] = value;
            }
        }
    }

    /*
     * PHASE 7: CALCULATE WEIGHT AND BIAS GRADIENTS
     *
     * No parameters are updated during this phase.
     */
    for (l = 0; l < layer_count; l++)
    {
        size_t in_size = network->layers[l].input_size;
        size_t out_size = network->layers[l].output_size;

        weight_grads[l] =
            matrix_create(in_size, out_size);

        bias_grads[l] =
            matrix_create(1, out_size);

        if (!matrix_is_valid(&weight_grads[l]) ||
            !matrix_is_valid(&bias_grads[l]))
        {
            goto cleanup;
        }

        /*
         * Weight gradient:
         *
         * dW = activation(transpose) * delta
         */
        for (i = 0; i < in_size; i++)
        {
            for (j = 0; j < out_size; j++)
            {
                double gradient = 0.0;

                for (r = 0; r < batch_size; r++)
                {
                    gradient +=
                        (double)activations[l].data[
                            r * in_size + i
                        ] *
                        deltas[l].data[
                            r * out_size + j
                        ];
                }

                float value = (float)gradient;

                if (!isfinite(value))
                    goto cleanup;

                weight_grads[l].data[
                    i * out_size + j
                ] = value;
            }
        }

        /*
         * Bias gradient:
         *
         * db = sum of deltas over batch rows
         */
        for (j = 0; j < out_size; j++)
        {
            double gradient = 0.0;

            for (r = 0; r < batch_size; r++)
            {
                gradient += deltas[l].data[
                    r * out_size + j
                ];
            }

            float value = (float)gradient;

            if (!isfinite(value))
                goto cleanup;

            bias_grads[l].data[j] = value;
        }
    }

    /*
     * PHASE 8: CALCULATE ONE GLOBAL GRADIENT NORM
     *
     * Combine every layer's weight and bias gradient norms.
     *
     * hypot() avoids unnecessary overflow when combining
     * large individual norms.
     */
    for (l = 0; l < layer_count; l++)
    {
        double weight_norm =
            gradient_global_norm(&weight_grads[l]);

        double bias_norm =
            gradient_global_norm(&bias_grads[l]);

        if (!isfinite(weight_norm) ||
            !isfinite(bias_norm))
        {
            goto cleanup;
        }

        global_norm = hypot(global_norm, weight_norm);
        global_norm = hypot(global_norm, bias_norm);

        if (!isfinite(global_norm))
            goto cleanup;
    }

    /*
     * Calculate one common scaling factor.
     *
     * All gradients retain their relative directions.
     */
    if (global_norm > (double)max_gradient_norm)
    {
        clip_scale =
            (double)max_gradient_norm / global_norm;
    }

    /*
     * PHASE 9: VALIDATE ALL PARAMETER UPDATES
     *
     * First check every candidate update.
     * Do not modify any network parameter yet.
     */
    for (l = 0; l < layer_count; l++)
    {
        size_t weight_count =
            network->layers[l].input_size *
            network->layers[l].output_size;

        size_t bias_count =
            network->layers[l].output_size;

        for (i = 0; i < weight_count; i++)
        {
            double gradient =
                (double)weight_grads[l].data[i] *
                clip_scale;

            double updated =
                (double)network->layers[l].weights.data[i] -
                (double)learning_rate * gradient;

            if (!isfinite(updated) ||
                !isfinite((float)updated))
            {
                goto cleanup;
            }
        }

        for (j = 0; j < bias_count; j++)
        {
            double gradient =
                (double)bias_grads[l].data[j] *
                clip_scale;

            double updated =
                (double)network->layers[l].biases.data[j] -
                (double)learning_rate * gradient;

            if (!isfinite(updated) ||
                !isfinite((float)updated))
            {
                goto cleanup;
            }
        }
    }

    /*
     * PHASE 10: APPLY ALL PARAMETER UPDATES
     *
     * Every candidate update has already been validated.
     */
    for (l = 0; l < layer_count; l++)
    {
        size_t weight_count =
            network->layers[l].input_size *
            network->layers[l].output_size;

        size_t bias_count =
            network->layers[l].output_size;

        for (i = 0; i < weight_count; i++)
        {
            double gradient =
                (double)weight_grads[l].data[i] *
                clip_scale;

            network->layers[l].weights.data[i] =
                (float)(
                    (double)network->layers[l].weights.data[i] -
                    (double)learning_rate * gradient
                );
        }

        for (j = 0; j < bias_count; j++)
        {
            double gradient =
                (double)bias_grads[l].data[j] *
                clip_scale;

            network->layers[l].biases.data[j] =
                (float)(
                    (double)network->layers[l].biases.data[j] -
                    (double)learning_rate * gradient
                );
        }
    }

    success = 1;

cleanup:

    /*
     * activations[0] aliases the caller's input.
     * Never free it.
     */
    if (activations != NULL)
    {
        for (l = 1; l <= layer_count; l++)
            matrix_free(&activations[l]);

        free(activations);
    }

    free_matrix_array(preactivations, layer_count);
    free_matrix_array(deltas, layer_count);
    free_matrix_array(weight_grads, layer_count);
    free_matrix_array(bias_grads, layer_count);

    matrix_free(&probabilities);

    return success ? loss : NAN;
}
