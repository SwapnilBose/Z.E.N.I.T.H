
#include <math.h>
#include <stddef.h>

#include "za_mlp.h"

/* Create a network with 1 input, 2 hidden neurons, 1 output. */
MLPNetwork mlp_create(void)
{
    MLPNetwork network;

    network.hidden = dense_create(1, 2);
    network.output = dense_create(2, 1);

    return network;
}

/* Free both layers. */
void mlp_free(MLPNetwork *network)
{
    if (network == NULL)
        return;

    dense_free(&network->hidden);
    dense_free(&network->output);
}

/*
 * Forward propagation:
 *
 * input -> hidden layer -> ReLU -> output layer
 */
Matrix mlp_forward(
    const MLPNetwork *network,
    const Matrix *input)
{
    if (network == NULL || input == NULL)
        return (Matrix){0};

    if (input->data == NULL ||
        network->hidden.weights.data == NULL ||
        network->output.weights.data == NULL ||
        input->cols != 1)
        return (Matrix){0};

    size_t count = input->rows;

    Matrix result = matrix_create(count, 1);

    if (result.data == NULL)
        return result;

    for (size_t i = 0; i < count; i++)
    {
        float x = input->data[i];

        float h1 = x * network->hidden.weights.data[0]
                     + network->hidden.biases.data[0];

        float h2 = x * network->hidden.weights.data[1]
                     + network->hidden.biases.data[1];

        /* ReLU activation */
        if (h1 < 0.0f) h1 = 0.0f;
        if (h2 < 0.0f) h2 = 0.0f;

        /* Output layer */
        result.data[i] =
            h1 * network->output.weights.data[0] +
            h2 * network->output.weights.data[1] +
            network->output.biases.data[0];
    }

    return result;
}

/*
 * Train the network using batch gradient descent.
 */
float mlp_train_step(
    MLPNetwork *network,
    const Matrix *input,
    const Matrix *target,
    float learning_rate)
{
    if (network == NULL || input == NULL || target == NULL)
        return NAN;

    if (input->data == NULL || target->data == NULL ||
        network->hidden.weights.data == NULL ||
        network->hidden.biases.data == NULL ||
        network->output.weights.data == NULL ||
        network->output.biases.data == NULL)
        return NAN;

    if (input->cols != 1 ||
        target->cols != 1 ||
        input->rows == 0 ||
        input->rows != target->rows ||
        !isfinite(learning_rate) ||
        learning_rate <= 0.0f)
        return NAN;

    size_t count = input->rows;

    /*
     * Gradients are accumulated here.
     * We update the weights only after processing the batch.
     */
    float gw1 = 0.0f;
    float gw2 = 0.0f;
    float gb1 = 0.0f;
    float gb2 = 0.0f;

    float gv1 = 0.0f;
    float gv2 = 0.0f;
    float gc = 0.0f;

    float loss = 0.0f;

    /*
     * Save the output weights before updating.
     * Backpropagation must use the weights that produced
     * the current predictions.
     */
    float v1 = network->output.weights.data[0];
    float v2 = network->output.weights.data[1];

    for (size_t i = 0; i < count; i++)
    {
        float x = input->data[i];
        float expected = target->data[i];

        /* ---------- FORWARD PASS ---------- */

        float z1 =
            x * network->hidden.weights.data[0] +
            network->hidden.biases.data[0];

        float z2 =
            x * network->hidden.weights.data[1] +
            network->hidden.biases.data[1];

        float h1 = z1 > 0.0f ? z1 : 0.0f;
        float h2 = z2 > 0.0f ? z2 : 0.0f;

        float predicted =
            h1 * v1 + h2 * v2 +
            network->output.biases.data[0];

        float error = predicted - expected;

        loss += error * error;

        /* ---------- BACKWARD PASS ---------- */

        /* Derivative of MSE with respect to prediction. */
        float d_output = 2.0f * error;

        /* Output-layer gradients. */
        gv1 += d_output * h1;
        gv2 += d_output * h2;
        gc  += d_output;

        /*
         * Propagate error backward through output weights.
         */
        float d_h1 = d_output * v1;
        float d_h2 = d_output * v2;

        /*
         * Apply the ReLU derivative.
         * If z <= 0, the gradient is zero.
         */
        float d_z1 = z1 > 0.0f ? d_h1 : 0.0f;
        float d_z2 = z2 > 0.0f ? d_h2 : 0.0f;

        /* Hidden-layer gradients. */
        gw1 += d_z1 * x;
        gw2 += d_z2 * x;

        gb1 += d_z1;
        gb2 += d_z2;
    }

    /* Average gradients across the batch. */
    float scale = 1.0f / (float)count;

    loss *= scale;

    gw1 *= scale;
    gw2 *= scale;
    gb1 *= scale;
    gb2 *= scale;

    gv1 *= scale;
    gv2 *= scale;
    gc  *= scale;

    /* ---------- PARAMETER UPDATE ---------- */

    float *hw = network->hidden.weights.data;
    float *hb = network->hidden.biases.data;

    float *ow = network->output.weights.data;
    float *ob = network->output.biases.data;

    hw[0] -= learning_rate * gw1;
    hw[1] -= learning_rate * gw2;

    hb[0] -= learning_rate * gb1;
    hb[1] -= learning_rate * gb2;

    ow[0] -= learning_rate * gv1;
    ow[1] -= learning_rate * gv2;

    ob[0] -= learning_rate * gc;

    return loss;
}
