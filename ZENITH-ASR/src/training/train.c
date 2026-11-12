
#include <math.h>
#include <stddef.h>

#include "za_train.h"


float dense_train_step(
    DenseLayer *layer,
    const Matrix *input,
    const Matrix *target,
    float learning_rate)
{
    /* Validate pointers and allocated memory. */
    if (layer == NULL || input == NULL || target == NULL)
    {
        return NAN;
    }

    if (input->data == NULL ||
        target->data == NULL ||
        layer->weights.data == NULL ||
        layer->biases.data == NULL)
    {
        return NAN;
    }

    /* This first trainer supports a single neuron. */
    if (layer->input_size != 1 ||
        layer->output_size != 1 ||
        input->cols != 1 ||
        target->cols != 1 ||
        input->rows == 0 ||
        input->rows != target->rows ||
        !isfinite(learning_rate) ||
        learning_rate <= 0.0f)
    {
        return NAN;
    }

    size_t count = input->rows;

    /*
        STEP 1: Forward propagation.

        Prediction = input * weight + bias

        No activation function because we are
        learning a linear relationship.
    */
    Matrix prediction = dense_forward(
        layer,
        input,
        ACTIVATION_NONE
    );

    if (prediction.data == NULL)
    {
        return NAN;
    }

    /*
        STEP 2: Calculate prediction errors
        and the MSE loss.
    */
    float loss = 0.0f;
    float gradient_weight = 0.0f;
    float gradient_bias = 0.0f;

    for (size_t i = 0; i < count; i++)
    {
        float x = input->data[i];
        float predicted = prediction.data[i];
        float expected = target->data[i];

        float error = predicted - expected;

        loss += error * error;

        /*
            Derivative of MSE with respect
            to the weight:

            (2/N) * error * input
        */
        gradient_weight += error * x;

        /*
            Derivative of MSE with respect
            to the bias:

            (2/N) * error
        */
        gradient_bias += error;
    }

    float scale = 2.0f / (float)count;

    loss /= (float)count;

    gradient_weight *= scale;
    gradient_bias *= scale;

    /*
        STEP 3: Gradient descent.

        New parameter = Old parameter - Learning rate
                        * Gradient
    */
    layer->weights.data[0] -=
        learning_rate * gradient_weight;

    layer->biases.data[0] -=
        learning_rate * gradient_bias;

    /* Release temporary prediction matrix. */
    matrix_free(&prediction);

    return loss;
}
