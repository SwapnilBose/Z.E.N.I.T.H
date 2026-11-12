
#include "za_network.h"

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
 * Check whether the network's layers have consistent dimensions.
 */
static int network_is_valid(const Network *network)
{
    size_t i;

    if (network == NULL ||
        network->layers == NULL ||
        network->layer_count == 0)
    {
        return 0;
    }

    for (i = 0; i < network->layer_count; i++)
    {
        const DenseLayer *layer = &network->layers[i];

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

        if (i > 0 &&
            network->layers[i - 1].output_size != layer->input_size)
        {
            return 0;
        }
    }

    return 1;
}


/*
 * Release an array of temporary matrices.
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
 * Create a dynamic neural network.
 *
 * sizes must contain layer_count + 1 entries.
 *
 * Example:
 * sizes = {2, 4, 3, 1}
 * layer_count = 3
 */
Network network_create(
    const size_t *sizes,
    size_t layer_count)
{
    Network network = {0};
    size_t i;

    if (sizes == NULL ||
        layer_count == 0 ||
        layer_count == SIZE_MAX ||
        layer_count > SIZE_MAX / sizeof(DenseLayer))
    {
        return network;
    }

    /* Every layer must contain at least one neuron. */
    for (i = 0; i <= layer_count; i++)
    {
        if (sizes[i] == 0)
            return network;
    }

    network.layers = calloc(layer_count, sizeof(DenseLayer));

    if (network.layers == NULL)
        return network;

    network.layer_count = layer_count;

    for (i = 0; i < layer_count; i++)
    {
        network.layers[i] =
            dense_create(sizes[i], sizes[i + 1]);

        /*
         * Verify that the dense layer was created successfully.
         */
        if (!matrix_is_valid(&network.layers[i].weights) ||
            !matrix_is_valid(&network.layers[i].biases))
        {
            network_free(&network);
            return (Network){0};
        }
    }

    return network;
}


/*
 * Free the entire network.
 */
void network_free(Network *network)
{
    size_t i;

    if (network == NULL)
        return;

    if (network->layers != NULL)
    {
        for (i = 0; i < network->layer_count; i++)
            dense_free(&network->layers[i]);

        free(network->layers);
    }

    network->layers = NULL;
    network->layer_count = 0;
}


/*
 * Forward propagation.
 *
 * All hidden layers use ReLU.
 * The final layer uses linear activation.
 */
Matrix network_forward(
    const Network *network,
    const Matrix *input)
{
    Matrix current = {0};
    size_t layer_index;

    if (!network_is_valid(network) ||
        !matrix_is_valid(input) ||
        input->cols != network->layers[0].input_size)
    {
        return current;
    }

    /*
     * Copy the input.
     * This prevents us from modifying the caller's matrix.
     */
    current = matrix_create(input->rows, input->cols);

    if (!matrix_is_valid(&current))
        return (Matrix){0};

    for (size_t i = 0; i < input->rows * input->cols; i++)
        current.data[i] = input->data[i];

    for (layer_index = 0;
         layer_index < network->layer_count;
         layer_index++)
    {
        const DenseLayer *layer = &network->layers[layer_index];

        Matrix next = matrix_create(
            current.rows,
            layer->output_size
        );

        if (!matrix_is_valid(&next))
        {
            matrix_free(&current);
            return (Matrix){0};
        }

        for (size_t r = 0; r < current.rows; r++)
        {
            for (size_t j = 0; j < layer->output_size; j++)
            {
                float sum = layer->biases.data[j];

                for (size_t i = 0; i < layer->input_size; i++)
                {
                    sum += current.data[r * current.cols + i] *
                           layer->weights.data[
                               i * layer->output_size + j
                           ];
                }

                /* ReLU on hidden layers only. */
                if (layer_index + 1 < network->layer_count &&
                    sum < 0.0f)
                {
                    sum = 0.0f;
                }

                next.data[r * next.cols + j] = sum;
            }
        }

        matrix_free(&current);
        current = next;
    }

    return current;
}


/*
 * One complete training step using batch gradient descent.
 *
 * The function calculates ALL deltas and gradients before
 * changing any weights or biases.
 */
float network_train_step(
    Network *network,
    const Matrix *input,
    const Matrix *target,
    float learning_rate)
{
    size_t layer_count;
    size_t batch_size;
    size_t output_size;
    size_t total_outputs;
    size_t l, r, i, j, k;

    Matrix *activations = NULL;
    Matrix *preactivations = NULL;
    Matrix *deltas = NULL;

    Matrix *weight_grads = NULL;
    Matrix *bias_grads = NULL;

    float loss = 0.0f;
    int success = 0;

    if (!network_is_valid(network) ||
        !matrix_is_valid(input) ||
        !matrix_is_valid(target) ||
        !isfinite(learning_rate) ||
        learning_rate <= 0.0f)
    {
        return NAN;
    }

    layer_count = network->layer_count;
    batch_size = input->rows;
    output_size =
        network->layers[layer_count - 1].output_size;

    if (input->cols != network->layers[0].input_size ||
        target->rows != batch_size ||
        target->cols != output_size)
    {
        return NAN;
    }

    if (batch_size > SIZE_MAX / output_size)
        return NAN;

    total_outputs = batch_size * output_size;

    /*
     * Allocate arrays for:
     *
     * activations:
     *     Network input and output of every layer.
     *
     * preactivations:
     *     Values before applying ReLU.
     *
     * deltas:
     *     Backpropagation errors for each layer.
     *
     * weight_grads / bias_grads:
     *     Calculated gradients.
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
     * The first activation is the caller's input.
     * It is borrowed, not owned by this function.
     */
    activations[0] = *input;

    /*
     * PHASE 1: FORWARD PROPAGATION
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
                float sum = layer->biases.data[j];

                for (i = 0; i < in_size; i++)
                {
                    sum += activations[l].data[
                               r * in_size + i
                           ] *
                           layer->weights.data[
                               i * out_size + j
                           ];
                }

                preactivations[l].data[
                    r * out_size + j
                ] = sum;

                /*
                 * Hidden layers: ReLU.
                 * Final layer: linear.
                 */
                if (l + 1 < layer_count && sum < 0.0f)
                    sum = 0.0f;

                activations[l + 1].data[
                    r * out_size + j
                ] = sum;
            }
        }
    }

    /*
     * PHASE 2: CALCULATE MSE LOSS
     */
    {
        const Matrix *prediction = &activations[layer_count];

        for (i = 0; i < total_outputs; i++)
        {
            float error = prediction->data[i] - target->data[i];

            loss += error * error;
        }

        loss /= (float)total_outputs;

        if (!isfinite(loss))
            goto cleanup;
    }

    /*
     * PHASE 3: OUTPUT DELTA
     *
     * MSE derivative:
     *
     * dL/dprediction = 2 * error / total_outputs
     */
    deltas[layer_count - 1] =
        matrix_create(batch_size, output_size);

    if (!matrix_is_valid(&deltas[layer_count - 1]))
        goto cleanup;

    for (i = 0; i < total_outputs; i++)
    {
        deltas[layer_count - 1].data[i] =
            2.0f *
            (activations[layer_count].data[i] - target->data[i]) /
            (float)total_outputs;
    }

    /*
     * PHASE 4: BACKPROPAGATION THROUGH HIDDEN LAYERS
     *
     * Calculate every delta before updating any weights.
     */
    if (layer_count > 1)
    {
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
                    float sum = 0.0f;

                    for (k = 0; k < next_size; k++)
                    {
                        sum += deltas[l + 1].data[
                                   r * next_size + k
                               ] *
                               network->layers[l + 1].weights.data[
                                   i * next_size + k
                               ];
                    }

                    /*
                     * ReLU derivative:
                     *
                     * z > 0: derivative = 1
                     * z <= 0: derivative = 0
                     */
                    if (preactivations[l].data[
                            r * current_size + i
                        ] <= 0.0f)
                    {
                        sum = 0.0f;
                    }

                    deltas[l].data[
                        r * current_size + i
                    ] = sum;
                }
            }
        }
    }

    /*
     * PHASE 5: CALCULATE ALL GRADIENTS
     *
     * Do not update weights yet.
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

        for (i = 0; i < in_size; i++)
        {
            for (j = 0; j < out_size; j++)
            {
                float gradient = 0.0f;

                for (r = 0; r < batch_size; r++)
                {
                    gradient +=
                        activations[l].data[
                            r * in_size + i
                        ] *
                        deltas[l].data[
                            r * out_size + j
                        ];
                }

                weight_grads[l].data[
                    i * out_size + j
                ] = gradient;
            }
        }

        for (j = 0; j < out_size; j++)
        {
            float gradient = 0.0f;

            for (r = 0; r < batch_size; r++)
            {
                gradient += deltas[l].data[
                    r * out_size + j
                ];
            }

            bias_grads[l].data[j] = gradient;
        }

        /*
         * Verify gradients before modifying parameters.
         */
        for (i = 0; i < in_size * out_size; i++)
        {
            if (!isfinite(weight_grads[l].data[i]))
                goto cleanup;
        }

        for (j = 0; j < out_size; j++)
        {
            if (!isfinite(bias_grads[l].data[j]))
                goto cleanup;
        }
    }

    /*
     * PHASE 6: UPDATE PARAMETERS
     *
     * All gradients were calculated using the original weights.
     */
    for (l = 0; l < layer_count; l++)
    {
        size_t in_size = network->layers[l].input_size;
        size_t out_size = network->layers[l].output_size;

        for (i = 0; i < in_size * out_size; i++)
        {
            float updated =
                network->layers[l].weights.data[i] -
                learning_rate * weight_grads[l].data[i];

            if (!isfinite(updated))
                goto cleanup;
        }

        for (j = 0; j < out_size; j++)
        {
            float updated =
                network->layers[l].biases.data[j] -
                learning_rate * bias_grads[l].data[j];

            if (!isfinite(updated))
                goto cleanup;
        }
    }

    /*
     * Apply the updates after validating them.
     */
    for (l = 0; l < layer_count; l++)
    {
        size_t in_size = network->layers[l].input_size;
        size_t out_size = network->layers[l].output_size;

        for (i = 0; i < in_size * out_size; i++)
        {
            network->layers[l].weights.data[i] -=
                learning_rate * weight_grads[l].data[i];
        }

        for (j = 0; j < out_size; j++)
        {
            network->layers[l].biases.data[j] -=
                learning_rate * bias_grads[l].data[j];
        }
    }

    success = 1;

cleanup:

    /* Do not free activations[0]: it aliases the caller's input. */
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

    return success ? loss : NAN;
}
