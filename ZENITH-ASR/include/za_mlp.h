
#ifndef ZA_MLP_H
#define ZA_MLP_H

#include "za_dense.h"

/*
 * A small two-layer neural network:
 *
 * Input -> Hidden layer (ReLU) -> Output layer (Linear)
 *
 * Hidden layer: 1 input, 2 neurons
 * Output layer: 2 inputs, 1 neuron
 */

typedef struct
{
    DenseLayer hidden;
    DenseLayer output;
} MLPNetwork;

/* Allocate and initialize the network. */
MLPNetwork mlp_create(void);

/* Release all network memory. */
void mlp_free(MLPNetwork *network);

/* Run a forward pass. Caller owns returned Matrix. */
Matrix mlp_forward(
    const MLPNetwork *network,
    const Matrix *input
);

/*
 * Perform one batch training step using backpropagation.
 * Returns the loss before updating weights, or NAN on failure.
 */
float mlp_train_step(
    MLPNetwork *network,
    const Matrix *input,
    const Matrix *target,
    float learning_rate
);

#endif
