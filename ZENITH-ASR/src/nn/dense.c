
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "za_dense.h"


/*
    Xavier uniform initialization.

    Random weights help prevent all neurons from
    starting with identical values.

    Limit = sqrt(6 / (input_size + output_size))
*/
static void initialize_weights(Matrix *weights,
                                size_t input_size,
                                size_t output_size)
{
    float limit = sqrtf(
        6.0f / (float)(input_size + output_size)
    );

    size_t count = weights->rows * weights->cols;

    for (size_t i = 0; i < count; i++)
    {
        float random_value =
            (float)rand() / (float)RAND_MAX;

        weights->data[i] =
            (2.0f * random_value - 1.0f) * limit;
    }
}


/*
    Create a dense layer.

    Weight dimensions:
        input_size x output_size

    Bias dimensions:
        1 x output_size
*/
DenseLayer dense_create(size_t input_size,
                        size_t output_size)
{
    DenseLayer layer = {0};

    if (input_size == 0 || output_size == 0)
    {
        return layer;
    }

    layer.weights =
        matrix_create(input_size, output_size);

    layer.biases =
        matrix_create(1, output_size);

    if (layer.weights.data == NULL ||
        layer.biases.data == NULL)
    {
        matrix_free(&layer.weights);
        matrix_free(&layer.biases);

        return layer;
    }

    layer.input_size = input_size;
    layer.output_size = output_size;

    /* Initialize weights; biases remain zero. */
    initialize_weights(
        &layer.weights,
        input_size,
        output_size
    );

    return layer;
}


/* Free all layer resources. */
void dense_free(DenseLayer *layer)
{
    if (layer == NULL)
    {
        return;
    }

    matrix_free(&layer->weights);
    matrix_free(&layer->biases);

    layer->input_size = 0;
    layer->output_size = 0;
}


/*
    Apply an activation function to one number.
*/
static float activate(float value, Activation activation)
{
    switch (activation)
    {
        case ACTIVATION_RELU:

            return value > 0.0f ? value : 0.0f;


        case ACTIVATION_SIGMOID:

            return 1.0f / (1.0f + expf(-value));


        case ACTIVATION_TANH:

            return tanhf(value);


        case ACTIVATION_NONE:
        default:

            return value;
    }
}


/*
    Forward propagation.

    Formula:

        output = activation(input * weights + biases)

    Supports multiple input rows (batch processing).
*/
Matrix dense_forward(const DenseLayer *layer,
                     const Matrix *input,
                     Activation activation)
{
    Matrix output = {0};

    if (layer == NULL || input == NULL)
    {
        return output;
    }

    if (layer->weights.data == NULL ||
        layer->biases.data == NULL ||
        input->data == NULL)
    {
        return output;
    }

    if (input->cols != layer->input_size)
    {
        return output;
    }

    /* Step 1: Multiply input by weights. */
    output = matrix_multiply(
        input,
        &layer->weights
    );

    if (output.data == NULL)
    {
        return output;
    }

    /*
        Step 2: Add biases.

        Each row receives the same bias values.
    */
    for (size_t i = 0; i < output.rows; i++)
    {
        for (size_t j = 0; j < output.cols; j++)
        {
            size_t index =
                i * output.cols + j;

            output.data[index] +=
                layer->biases.data[j];
        }
    }

    /* Step 3: Apply activation function. */
    size_t count = output.rows * output.cols;

    for (size_t i = 0; i < count; i++)
    {
        output.data[i] = activate(
            output.data[i],
            activation
        );
    }

    return output;
}
