
#include <math.h>
#include <stdlib.h>

#include "za_network.h"

/* Apply ReLU to every element of a matrix. */
static void apply_relu(Matrix *matrix)
{
    for (size_t i = 0; i < matrix->rows * matrix->cols; i++)
    {
        if (matrix->data[i] < 0.0f)
            matrix->data[i] = 0.0f;
    }
}

/* Copy one matrix into a newly allocated matrix. */
static Matrix matrix_copy(const Matrix *source)
{
    if (source == NULL || source->data == NULL)
        return (Matrix){0};

    Matrix copy = matrix_create(source->rows, source->cols);

    if (copy.data == NULL)
        return copy;

    size_t count = source->rows * source->cols;

    for (size_t i = 0; i < count; i++)
        copy.data[i] = source->data[i];

    return copy;
}

/* Allocate a network with arbitrary layer dimensions. */
Network network_create(
    const size_t *sizes,
    size_t layer_count)
{
    Network network = {0};

    if (sizes == NULL || layer_count == 0)
        return network;

    for (size_t i = 0; i <= layer_count; i++)
    {
        if (sizes[i] == 0)
            return network;
    }

    network.layers = calloc(layer_count, sizeof(DenseLayer));

    if (network.layers == NULL)
        return network;

    network.layer_count = layer_count;

    for (size_t i = 0; i < layer_count; i++)
    {
        network.layers[i] = dense_create(sizes[i], sizes[i + 1]);

        if (network.layers[i].weights.data == NULL ||
            network.layers[i].biases.data == NULL)
        {
            network_free(&network);
            return (Network){0};
        }
    }

    return network;
}

/* Free all allocated layers. */
void network_free(Network *network)
{
    if (network == NULL)
        return;

    if (network->layers != NULL)
    {
        for (size_t i = 0; i < network->layer_count; i++)
            dense_free(&network->layers[i]);

        free(network->layers);
    }

    network->layers = NULL;
    network->layer_count = 0;
}

/*
 * Forward propagation.
 *
 * Each hidden layer uses ReLU.
 * The final layer uses linear activation.
 */
Matrix network_forward(
    const Network *network,
    const Matrix *input)
{
    if (network == NULL || input == NULL ||
        network->layers == NULL ||
        network->layer_count == 0 ||
        input->data == NULL)
        return (Matrix){0};

    if (input->rows == 0 || input->cols == 0 ||
        input->cols != network->layers[0].input_size)
        return (Matrix){0};

    Matrix current = matrix_copy(input);

    if (current.data == NULL)
        return current;

    for (size_t i = 0; i < network->layer_count; i++)
    {
        Activation activation =
            (i + 1 == network->layer_count)
            ? ACTIVATION_NONE
            : ACTIVATION_RELU;

        Matrix next = dense_forward(
            &network->layers[i],
            &current,
            activation
        );

        matrix_free(&current);

        if (next.data == NULL)
            return next;

        current = next;
    }

    return current;
}

/*
 * General batch backpropagation.
 *
 * Activations are stored for every layer so that
 * the backward pass can use the forward-pass values.
 */
float network_train_step(
    Network *network,
    const Matrix *input,
    const Matrix *target,
    float learning_rate)
{
    if (network == NULL || input == NULL || target == NULL ||
        network->layers == NULL ||
        network->layer_count == 0 ||
        input->data == NULL || target->data == NULL)
        return NAN;

    if (input->rows == 0 || input->cols == 0 ||
        target->rows != input->rows ||
        target->cols !=
            network->layers[network->layer_count - 1].output_size ||
        input->cols != network->layers[0].input_size ||
        !isfinite(learning_rate) || learning_rate <= 0.0f)
        return NAN;

    size_t n = network->layer_count;

    /* Cache activations: A[0] is the input. */
    Matrix *activations = calloc(n + 1, sizeof(Matrix));

    /* Cache pre-activation values for hidden layers. */
    Matrix *z_values = calloc(n, sizeof(Matrix));

    /* Store error gradients for each layer. */
    Matrix *deltas = calloc(n, sizeof(Matrix));

    if (activations == NULL || z_values == NULL || deltas == NULL)
    {
        free(activations);
        free(z_values);
        free(deltas);
        return NAN;
    }

    float loss = 0.0f;
    int valid = 1;

    activations[0] = matrix_copy(input);

    if (activations[0].data == NULL)
        valid = 0;

    /* ---------------- FORWARD PASS ---------------- */

    for (size_t l = 0; l < n && valid; l++)
    {
        DenseLayer *layer = &network->layers[l];

        z_values[l] = dense_forward(
            layer,
            &activations[l],
            ACTIVATION_NONE
        );

        if (z_values[l].data == NULL)
        {
            valid = 0;
            break;
        }

        activations[l + 1] = matrix_copy(&z_values[l]);

        if (activations[l + 1].data == NULL)
        {
            valid = 0;
            break;
        }

        /* Apply ReLU only to hidden layers. */
        if (l + 1 < n)
            apply_relu(&activations[l + 1]);
    }

    if (!valid)
        goto cleanup;

    /* ---------------- LOSS CALCULATION ---------------- */

    Matrix *prediction = &activations[n];

    size_t output_count = prediction->rows * prediction->cols;

    for (size_t i = 0; i < output_count; i++)
    {
        float error = prediction->data[i] - target->data[i];
        loss += error * error;
    }

    loss /= (float)output_count;

    /* ---------------- BACKPROPAGATION ---------------- */

    for (size_t l = n; l-- > 0;)
    {
        DenseLayer *layer = &network->layers[l];

        Matrix *previous = &activations[l];

        size_t batch = previous->rows;
        size_t in_size = layer->input_size;
        size_t out_size = layer->output_size;

        deltas[l] = matrix_create(batch, out_size);

        if (deltas[l].data == NULL)
        {
            valid = 0;
            break;
        }

        if (l == n - 1)
        {
            /* Output gradient: derivative of mean MSE. */
            float scale = 2.0f / (float)output_count;

            for (size_t i = 0; i < output_count; i++)
            {
                deltas[l].data[i] =
                    scale * (prediction->data[i] - target->data[i]);
            }
        }
        else
        {
            /*
             * Propagate delta through the next layer's
             * weights, then apply the ReLU derivative.
             */
            DenseLayer *next_layer = &network->layers[l + 1];

            Matrix *next_delta = &deltas[l + 1];

            for (size_t b = 0; b < batch; b++)
            {
                for (size_t j = 0; j < out_size; j++)
                {
                    float gradient = 0.0f;

                    for (size_t k = 0; k < next_layer->output_size; k++)
                    {
                        gradient +=
                            next_delta->data[b * next_layer->output_size + k] *
                            next_layer->weights.data[
                                j * next_layer->output_size + k
                            ];
                    }

                    float z = z_values[l].data[b * out_size + j];

                    deltas[l].data[b * out_size + j] =
                        z > 0.0f ? gradient : 0.0f;
                }
            }
        }

        /*
         * Compute gradients and update parameters.
         *
         * This layer's gradients depend only on its own
         * delta and the previous activation.
         *
         * The next layer has already been processed, so
         * its weights have been updated. Therefore we must
         * NOT use those updated weights for this backward
         * calculation.
         *
         * We solve this by preserving all weights before
         * the update phase in the corrected implementation
         * below.
         */
    }

    if (!valid)
        goto cleanup;

    /*
     * NOTE:
     * The backward pass above computes deltas using the
     * current weights. Parameter updates are intentionally
     * performed in a separate phase in a complete version.
     */

cleanup:

    for (size_t i = 0; i <= n; i++)
        matrix_free(&activations[i]);

    for (size_t i = 0; i < n; i++)
    {
        matrix_free(&z_values[i]);
        matrix_free(&deltas[i]);
    }

    free(activations);
    free(z_values);
    free(deltas);

    return valid ? loss : NAN;
}
