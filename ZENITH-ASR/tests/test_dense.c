
#include <stdio.h>
#include <stdlib.h>

#include "za_dense.h"

int main(void)
{
    /* Make random initialization reproducible. */
    srand(42);

    /*
        Create a layer:

        2 input values
        2 output neurons
    */
    DenseLayer layer = dense_create(2, 2);

    if (layer.weights.data == NULL ||
        layer.biases.data == NULL)
    {
        printf("Layer allocation failed.\n");
        dense_free(&layer);
        return 1;
    }

    /*
        Set known weights manually so that
        we can verify the mathematics.

        W =  0.5  -1.0
             1.0   0.5
    */
    layer.weights.data[0] = 0.5f;
    layer.weights.data[1] = -1.0f;
    layer.weights.data[2] = 1.0f;
    layer.weights.data[3] = 0.5f;

    /*
        Biases:

        b = 0.1  -0.2
    */
    layer.biases.data[0] = 0.1f;
    layer.biases.data[1] = -0.2f;

    /* Input: [1, 2] */
    Matrix input = matrix_create(1, 2);

    if (input.data == NULL)
    {
        printf("Input allocation failed.\n");
        dense_free(&layer);
        return 1;
    }

    input.data[0] = 1.0f;
    input.data[1] = 2.0f;

    printf("Input:\n");
    matrix_print(&input);

    printf("\nWeights:\n");
    matrix_print(&layer.weights);

    printf("\nBiases:\n");
    matrix_print(&layer.biases);

    /* Perform forward propagation. */
    Matrix output = dense_forward(
        &layer,
        &input,
        ACTIVATION_RELU
    );

    if (output.data == NULL)
    {
        printf("Forward propagation failed.\n");

        matrix_free(&input);
        dense_free(&layer);

        return 1;
    }

    printf("\nOutput after ReLU:\n");
    matrix_print(&output);

    /*
        Expected result:

        2.600   0.000
    */

    /* Release allocated memory. */
    matrix_free(&input);
    matrix_free(&output);
    dense_free(&layer);

    return 0;
}
