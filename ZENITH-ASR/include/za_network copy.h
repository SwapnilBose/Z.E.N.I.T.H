
#ifndef ZA_NETWORK_H
#define ZA_NETWORK_H

#include <stddef.h>
#include "za_dense.h"

typedef struct
{
    DenseLayer *layers;
    size_t layer_count;
} Network;

/*
 * sizes contains the neuron count of each layer,
 * including the input and output dimensions.
 *
 * Example:
 * sizes = {2, 4, 3, 1}
 * layer_count = 3
 */
Network network_create(
    const size_t *sizes,
    size_t layer_count
);

/* Release all network layers. */
void network_free(Network *network);

/* Forward propagation through every layer. */
Matrix network_forward(
    const Network *network,
    const Matrix *input
);

/* One batch gradient-descent training step. */
float network_train_step(
    Network *network,
    const Matrix *input,
    const Matrix *target,
    float learning_rate
);

#endif
