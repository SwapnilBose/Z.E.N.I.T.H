
#ifndef ZA_NETWORK_H
#define ZA_NETWORK_H

#include <stddef.h>
#include "za_dense.h"

/*
 * A dynamic neural network.
 *
 * layers:
 *     Dynamically allocated array of DenseLayer structures.
 *
 * layer_count:
 *     Number of trainable layers.
 *
 * The input layer is NOT counted as a trainable layer.
 */
typedef struct {
    DenseLayer *layers;
    size_t layer_count;
} Network;


/*
 * Creates a neural network.
 *
 * sizes contains the number of neurons in each layer.
 *
 * Example:
 *     {2, 4, 3, 1}
 *
 * Architecture:
 *     2 inputs
 *     4 neurons in hidden layer 1
 *     3 neurons in hidden layer 2
 *     1 output
 *
 * layer_count = 3 because there are three connections
 * between adjacent layers.
 */
Network network_create(
    const size_t *sizes,
    size_t layer_count
);


/* Frees all layers and resets the network. */
void network_free(Network *network);


/*
 * Performs forward propagation.
 *
 * Hidden layers use ReLU.
 * Final layer uses linear activation.
 *
 * Returns a newly allocated output matrix.
 */
Matrix network_forward(
    const Network *network,
    const Matrix *input
);


/*
 * Performs one complete training step:
 *
 * 1. Forward propagation
 * 2. Loss calculation
 * 3. Backpropagation
 * 4. Gradient calculation
 * 5. Parameter updates
 *
 * Supports batches.
 *
 * Returns the MSE calculated BEFORE the update.
 * Returns NAN if the operation fails.
 */
float network_train_step(
    Network *network,
    const Matrix *input,
    const Matrix *target,
    float learning_rate
);

/*
 * Performs one classification training step.
 *
 * The final layer produces logits.
 * Softmax converts logits into probabilities.
 * Cross-entropy calculates the loss.
 *
 * Targets must be one-hot encoded.
 *
 * max_gradient_norm controls global gradient clipping
 * across all weights and biases in the network.
 *
 * Returns the mean cross-entropy loss before the update,
 * or NAN if the operation fails.
 */
float network_train_classification_step(
    Network *network,
    const Matrix *input,
    const Matrix *one_hot_targets,
    float learning_rate,
    float max_gradient_norm
);

#endif
