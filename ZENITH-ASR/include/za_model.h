
#ifndef ZA_MODEL_H
#define ZA_MODEL_H

#include "za_network.h"
#include "za_normalize.h"

/*
 * Complete ZENITH model.
 *
 * A model contains:
 *
 * 1. Neural network architecture.
 * 2. Trained weights and biases.
 * 3. Input normalization statistics.
 *
 * The model owns all memory inside these structures.
 */
typedef struct
{
    Network network;
    Normalizer normalizer;

} ZAModel;


/*
 * Save a trained network and its normalizer.
 *
 * filename:
 *     Destination binary model file.
 *
 * network:
 *     Trained neural network.
 *
 * normalizer:
 *     Normalizer fitted on the training dataset.
 *
 * Returns:
 *     1 on success.
 *     0 on failure.
 */
int model_save(
    const char *filename,
    const Network *network,
    const Normalizer *normalizer
);


/*
 * Load a model from a binary file.
 *
 * Returns:
 *     A populated ZAModel on success.
 *     An empty ZAModel on failure.
 *
 * The returned model owns its allocated memory.
 */
ZAModel model_load(const char *filename);


/*
 * Predict using raw, unnormalized input.
 *
 * Automatically:
 *     1. Applies the saved normalizer.
 *     2. Performs neural network forward propagation.
 *
 * Returns a newly allocated output Matrix.
 *
 * Returns an empty Matrix on failure.
 */
Matrix model_predict(
    const ZAModel *model,
    const Matrix *input
);


/* Release the network and normalizer owned by a model. */
void model_free(ZAModel *model);


/* Check whether a model has valid dimensions and storage. */
int model_is_valid(const ZAModel *model);

#endif
