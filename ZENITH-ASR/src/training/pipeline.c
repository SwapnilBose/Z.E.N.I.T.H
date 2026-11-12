#include "za_pipeline.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * Check that a matrix has allocated storage, non-zero dimensions,
 * safe dimensions, and only finite values.
 */
static int matrix_is_finite_and_valid(const Matrix *matrix)
{
    if (matrix == NULL ||
        matrix->data == NULL ||
        matrix->rows == 0 ||
        matrix->cols == 0 ||
        matrix->rows > SIZE_MAX / matrix->cols)
    {
        return 0;
    }

    size_t count = matrix->rows * matrix->cols;

    for (size_t i = 0; i < count; i++)
    {
        if (!isfinite(matrix->data[i]))
            return 0;
    }

    return 1;
}

/*
 * Check that Dataset metadata agrees with its two matrices.
 */
static int dataset_is_valid(const Dataset *dataset)
{
    if (dataset == NULL ||
        dataset->sample_count == 0 ||
        dataset->input_size == 0 ||
        dataset->output_size == 0 ||
        !matrix_is_finite_and_valid(&dataset->inputs) ||
        !matrix_is_finite_and_valid(&dataset->targets))
    {
        return 0;
    }

    return dataset->inputs.rows == dataset->sample_count &&
           dataset->inputs.cols == dataset->input_size &&
           dataset->targets.rows == dataset->sample_count &&
           dataset->targets.cols == dataset->output_size;
}

/*
 * Check that the network has usable input and output layers.
 *
 * The network implementation performs its own deeper validation.
 * These checks ensure that the datasets agree with the network.
 */
static int network_matches_dataset(
    const Network *network,
    const Dataset *training)
{
    if (network == NULL ||
        network->layers == NULL ||
        network->layer_count == 0)
    {
        return 0;
    }

    const DenseLayer *first = &network->layers[0];
    const DenseLayer *last =
        &network->layers[network->layer_count - 1];

    if (first->input_size != training->input_size ||
        last->output_size != training->output_size)
    {
        return 0;
    }

    return 1;
}

/*
 * A Normalizer is considered empty only when all fields are zero/null.
 */
static int normalizer_is_empty(const Normalizer *normalizer)
{
    return normalizer != NULL &&
           normalizer->means == NULL &&
           normalizer->scales == NULL &&
           normalizer->feature_count == 0;
}

/*
 * Initialize the deterministic pseudo-random generator.
 * A zero state is replaced because xorshift cannot escape zero.
 */
static uint64_t random_seed(uint64_t seed)
{
    if (seed == 0)
        return UINT64_C(0x9E3779B97F4A7C15);

    return seed;
}

/* xorshift64* pseudo-random number generator. */
static uint64_t random_next(uint64_t *state)
{
    uint64_t value = *state;

    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;

    *state = value;

    return value * UINT64_C(2685821657736338717);
}

/*
 * Return a random integer in [0, bound).
 * Rejection sampling avoids modulo bias.
 */
static size_t random_bounded(uint64_t *state, size_t bound)
{
    uint64_t limit = (uint64_t)bound;
    uint64_t threshold = (uint64_t)(0 - limit) % limit;
    uint64_t value;

    do
    {
        value = random_next(state);
    }
    while (value < threshold);

    return (size_t)(value % limit);
}

/* Shuffle sample indices using Fisher-Yates. */
static void shuffle_indices(size_t *indices, size_t count, uint64_t *state)
{
    for (size_t i = count; i > 1; i--)
    {
        size_t j = random_bounded(state, i);
        size_t temporary = indices[i - 1];

        indices[i - 1] = indices[j];
        indices[j] = temporary;
    }
}

/*
 * Copy selected rows into a temporary mini-batch.
 * Input and target rows use the same indices, preserving their pairing.
 */
static int create_batch(
    const Matrix *all_inputs,
    const Matrix *all_targets,
    const size_t *indices,
    size_t batch_start,
    size_t batch_count,
    Matrix *batch_inputs,
    Matrix *batch_targets)
{
    *batch_inputs = matrix_create(batch_count, all_inputs->cols);

    if (batch_inputs->data == NULL)
        return 0;

    *batch_targets = matrix_create(batch_count, all_targets->cols);

    if (batch_targets->data == NULL)
    {
        matrix_free(batch_inputs);
        return 0;
    }

    for (size_t row = 0; row < batch_count; row++)
    {
        size_t source_row = indices[batch_start + row];

        for (size_t col = 0; col < all_inputs->cols; col++)
        {
            batch_inputs->data[row * all_inputs->cols + col] =
                all_inputs->data[source_row * all_inputs->cols + col];
        }

        for (size_t col = 0; col < all_targets->cols; col++)
        {
            batch_targets->data[row * all_targets->cols + col] =
                all_targets->data[source_row * all_targets->cols + col];
        }
    }

    return 1;
}

/*
 * Calculate mean squared error between two equally sized matrices.
 */
static float calculate_mse(const Matrix *predictions, const Matrix *targets)
{
    if (!matrix_is_finite_and_valid(predictions) ||
        !matrix_is_finite_and_valid(targets) ||
        predictions->rows != targets->rows ||
        predictions->cols != targets->cols)
    {
        return NAN;
    }

    size_t count = predictions->rows * predictions->cols;
    double sum = 0.0;

    for (size_t i = 0; i < count; i++)
    {
        double difference =
            (double)predictions->data[i] - targets->data[i];

        sum += difference * difference;
    }

    double mse = sum / (double)count;

    if (!isfinite(mse) || mse > (double)HUGE_VAL)
        return NAN;

    float result = (float)mse;

    if (!isfinite(result))
        return NAN;

    return result;
}

/*
 * Initialize a TrainingPipeline.
 *
 * Call this only for a new/empty pipeline. Use
 * training_pipeline_free() before reinitializing an existing one.
 */
void training_pipeline_init(TrainingPipeline *pipeline)
{
    if (pipeline == NULL)
        return;

    pipeline->input_normalizer = (Normalizer){0};
}

/* Release the normalizer owned by a pipeline. */
void training_pipeline_free(TrainingPipeline *pipeline)
{
    if (pipeline == NULL)
        return;

    normalizer_free(&pipeline->input_normalizer);
}

/* Release both arrays owned by a TrainingHistory. */
void training_history_free(TrainingHistory *history)
{
    if (history == NULL)
        return;

    free(history->training_loss);
    free(history->validation_loss);

    history->training_loss = NULL;
    history->validation_loss = NULL;
    history->epochs_completed = 0;
}

/*
 * Evaluate already-normalized inputs without changing network weights.
 * This helper avoids repeatedly normalizing validation data every epoch.
 */
static float evaluate_normalized(
    const Network *network,
    const Matrix *normalized_inputs,
    const Matrix *targets)
{
    Matrix predictions = {0};
    float loss = NAN;

    if (!matrix_is_finite_and_valid(normalized_inputs) ||
        !matrix_is_finite_and_valid(targets) ||
        normalized_inputs->rows != targets->rows)
    {
        return NAN;
    }

    predictions = network_forward(network, normalized_inputs);

    if (predictions.data == NULL)
        return NAN;

    loss = calculate_mse(&predictions, targets);
    matrix_free(&predictions);

    return loss;
}

/*
 * Evaluate a dataset without changing network weights.
 *
 * The same training-fitted normalizer is used for every dataset.
 */
float training_pipeline_evaluate(
    const TrainingPipeline *pipeline,
    const Network *network,
    const Dataset *dataset)
{
    Matrix normalized_inputs = {0};
    float loss = NAN;

    if (pipeline == NULL ||
        !dataset_is_valid(dataset) ||
        !network_matches_dataset(network, dataset) ||
        pipeline->input_normalizer.means == NULL ||
        pipeline->input_normalizer.scales == NULL ||
        pipeline->input_normalizer.feature_count != dataset->input_size)
    {
        return NAN;
    }

    if (!normalizer_transform(
            &pipeline->input_normalizer,
            &dataset->inputs,
            &normalized_inputs))
    {
        return NAN;
    }

    loss = evaluate_normalized(
        network,
        &normalized_inputs,
        &dataset->targets);

    matrix_free(&normalized_inputs);

    return loss;
}

/*
 * Train a network using mini-batches.
 *
 * Normalization statistics are fitted ONLY on training inputs.
 * Validation inputs are transformed using those same statistics.
 */
int training_pipeline_train(
    TrainingPipeline *pipeline,
    Network *network,
    const Dataset *training,
    const Dataset *validation,
    const TrainingConfig *config,
    TrainingHistory *history)
{
    Normalizer fitted_normalizer = {0};

    Matrix normalized_training_inputs = {0};
    Matrix normalized_validation_inputs = {0};

    size_t *indices = NULL;
    float *training_losses = NULL;
    float *validation_losses = NULL;

    uint64_t random_state;
    int success = 0;

    if (pipeline == NULL ||
        history == NULL ||
        config == NULL ||
        !normalizer_is_empty(&pipeline->input_normalizer) ||
        history->training_loss != NULL ||
        history->validation_loss != NULL ||
        history->epochs_completed != 0)
    {
        return 0;
    }

    if (!dataset_is_valid(training) ||
        !dataset_is_valid(validation) ||
        !network_matches_dataset(network, training) ||
        training->input_size != validation->input_size ||
        training->output_size != validation->output_size ||
        config->epochs == 0 ||
        config->batch_size == 0 ||
        !isfinite(config->learning_rate) ||
        config->learning_rate <= 0.0f)
    {
        return 0;
    }

    /* Check allocation-size arithmetic before allocating arrays. */
    if (config->epochs > SIZE_MAX / sizeof(float) ||
        training->sample_count > SIZE_MAX / sizeof(size_t))
    {
        return 0;
    }

    /*
     * STEP 1:
     * Fit normalization statistics using TRAINING INPUTS ONLY.
     */
    fitted_normalizer = normalizer_fit(&training->inputs);

    if (fitted_normalizer.means == NULL ||
        fitted_normalizer.scales == NULL)
    {
        goto cleanup;
    }

    /*
     * STEP 2:
     * Transform both datasets using the training normalizer.
     *
     * The original Dataset matrices remain unchanged.
     */
    if (!normalizer_transform(
            &fitted_normalizer,
            &training->inputs,
            &normalized_training_inputs))
    {
        goto cleanup;
    }

    if (!normalizer_transform(
            &fitted_normalizer,
            &validation->inputs,
            &normalized_validation_inputs))
    {
        goto cleanup;
    }

    /*
     * STEP 3:
     * Allocate sample indices and one loss value per epoch.
     */
    indices = malloc(training->sample_count * sizeof(size_t));
    training_losses = calloc(config->epochs, sizeof(float));
    validation_losses = calloc(config->epochs, sizeof(float));

    if (indices == NULL ||
        training_losses == NULL ||
        validation_losses == NULL)
    {
        goto cleanup;
    }

    for (size_t i = 0; i < training->sample_count; i++)
        indices[i] = i;

    random_state = random_seed(config->shuffle_seed);

    /*
     * STEP 4:
     * Training loop.
     */
    for (size_t epoch = 0; epoch < config->epochs; epoch++)
    {
        double epoch_loss_sum = 0.0;

        /* Re-shuffle the training order at the start of each epoch. */
        shuffle_indices(indices, training->sample_count, &random_state);

        /*
         * Process the dataset in mini-batches.
         * The final batch may contain fewer samples.
         */
        for (size_t start = 0;
             start < training->sample_count;
             start += config->batch_size)
        {
            size_t remaining = training->sample_count - start;
            size_t current_batch_size =
                remaining < config->batch_size
                    ? remaining
                    : config->batch_size;

            Matrix batch_inputs = {0};
            Matrix batch_targets = {0};

            if (!create_batch(
                    &normalized_training_inputs,
                    &training->targets,
                    indices,
                    start,
                    current_batch_size,
                    &batch_inputs,
                    &batch_targets))
            {
                matrix_free(&batch_inputs);
                matrix_free(&batch_targets);
                goto cleanup;
            }

            /*
             * network_train_step returns the MSE BEFORE updating
             * parameters. Its batch loss is weighted by batch size
             * when computing the epoch average.
             */
            float batch_loss = network_train_step(
                network,
                &batch_inputs,
                &batch_targets,
                config->learning_rate);

            matrix_free(&batch_inputs);
            matrix_free(&batch_targets);

            if (!isfinite(batch_loss))
                goto cleanup;

            epoch_loss_sum +=
                (double)batch_loss * (double)current_batch_size;
        }

        training_losses[epoch] =
            (float)(epoch_loss_sum / (double)training->sample_count);

        if (!isfinite(training_losses[epoch]))
            goto cleanup;

        /*
         * STEP 5:
         * Evaluate the pre-normalized validation data after the epoch.
         * This performs forward propagation only; no weights change.
         */
        validation_losses[epoch] = evaluate_normalized(
            network,
            &normalized_validation_inputs,
            &validation->targets);
        if (!isfinite(validation_losses[epoch]))
            goto cleanup;

        history->epochs_completed = epoch + 1;
    }

    /*
     * Transfer ownership of the fitted normalizer and history arrays
     * to the caller-facing structures.
     */
    pipeline->input_normalizer = fitted_normalizer;
    fitted_normalizer = (Normalizer){0};

    history->training_loss = training_losses;
    history->validation_loss = validation_losses;

    training_losses = NULL;
    validation_losses = NULL;

    success = 1;

cleanup:

    normalizer_free(&fitted_normalizer);

    matrix_free(&normalized_training_inputs);
    matrix_free(&normalized_validation_inputs);

    free(indices);
    free(training_losses);
    free(validation_losses);

    if (!success)
    {
        training_pipeline_free(pipeline);
        training_history_free(history);
    }

    return success;
}
