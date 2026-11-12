
#ifndef ZA_DENSE_H
#define ZA_DENSE_H

#include "za_matrix.h"

/*
    Supported activation functions.
*/
typedef enum
{
    ACTIVATION_NONE,
    ACTIVATION_RELU,
    ACTIVATION_SIGMOID,
    ACTIVATION_TANH

} Activation;


/*
    Dense neural-network layer.

    weights:
        Connection strengths between inputs and neurons.

    biases:
        Additional adjustable values for each neuron.

    input_size:
        Number of inputs.

    output_size:
        Number of neurons.
*/
typedef struct
{
    Matrix weights;
    Matrix biases;

    size_t input_size;
    size_t output_size;

} DenseLayer;


/* Create and initialize a dense layer. */
DenseLayer dense_create(size_t input_size,
                        size_t output_size);

/* Release layer memory. */
void dense_free(DenseLayer *layer);

/* Calculate the layer's forward pass. */
Matrix dense_forward(const DenseLayer *layer,
                     const Matrix *input,
                     Activation activation);

#endif
