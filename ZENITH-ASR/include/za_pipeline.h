#ifndef ZA_PIPELINE_H
#define ZA_PIPELINE_H

#include <stddef.h>
#include <stdint.h>

#include "za_dataset.h"
#include "za_network.h"
#include "za_normalize.h"

/*
 * Configuration for one complete network-training run.
 */
typedef struct
{
    size_t epochs;
    size_t batch_size;

    /*
     * Gradient-descent learning rate.
     * Must be finite and greater than zero.
     */
    float learning_rate;

    /*
     * Seed used to shuffle training samples.
     * The same seed should reproduce the same shuffle sequence.
     */
    uint64_t shuffle_seed;
} TrainingConfig;

/*
 * Loss values recorded during training.
 *
 * training_loss[e]   = average training loss for epoch e
 * validation_loss[e] = validation loss after epoch e
 *
 * Both arrays contain epochs_completed elements.
 */
typedef struct
{
    float *training_loss;
    float *validation_loss;

    size_t epochs_completed;
} TrainingHistory;

/*
 * State retained by the training pipeline.
 *
 * The fitted normalizer is learned from training inputs only.
 * It must also be used to normalize validation, test, and
 * future inference inputs.
 */
typedef struct
{
    Normalizer input_normalizer;
} TrainingPipeline;

/*
 * Initialize a pipeline to its empty state.
 */
void training_pipeline_init(TrainingPipeline *pipeline);

/*
 * Train an existing network using training and validation datasets.
 *
 * The function:
 * 1. Fits the input normalizer using training inputs only.
 * 2. Normalizes training and validation inputs with that normalizer.
 * 3. Trains the network using mini-batches.
 * 4. Evaluates validation loss without updating network weights.
 * 5. Records both losses for every completed epoch.
 *
 * Original Dataset objects are never modified.
 *
 * On success:
 * - Returns 1.
 * - pipeline owns the fitted normalizer.
 * - history owns its allocated loss arrays.
 *
 * On failure:
 * - Returns 0.
 * - Any partially created resources are released.
 * - pipeline and history are reset to empty states.
 *
 * The caller must pass an initialized/empty pipeline and history.
 */
int training_pipeline_train(
    TrainingPipeline *pipeline,
    Network *network,
    const Dataset *training,
    const Dataset *validation,
    const TrainingConfig *config,
    TrainingHistory *history
);

/*
 * Evaluate a dataset using the trained network and the normalizer
 * fitted during training.
 *
 * This function does not update network parameters.
 *
 * Returns mean squared error (MSE).
 * Returns NAN if evaluation fails.
 */
float training_pipeline_evaluate(
    const TrainingPipeline *pipeline,
    const Network *network,
    const Dataset *dataset
);

/*
 * Release the normalizer owned by the pipeline.
 */
void training_pipeline_free(TrainingPipeline *pipeline);

/*
 * Release the loss arrays owned by TrainingHistory.
 */
void training_history_free(TrainingHistory *history);

#endif
