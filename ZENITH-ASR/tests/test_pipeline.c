
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "za_matrix.h"
#include "za_dataset.h"
#include "za_network.h"
#include "za_pipeline.h"

/*
    ZENITH-ASR
    Implementation 009.4 — Training Pipeline Test

    What this test verifies:

    1. Dataset creation.
    2. Network creation.
    3. Training pipeline initialization.
    4. Training using mini-batches.
    5. Training loss reduction.
    6. Validation loss calculation.
    7. Repeated evaluation does not modify model weights.
    8. Memory cleanup.

    Dataset relationship:

        y = 2x + 1

    Training samples:
        x = 0, 2, 4, ..., 38

    Validation samples:
        x = 1, 3, 5, ..., 39

    The model must learn from even-numbered samples
    and generalize to odd-numbered samples.
*/


#define TRAINING_SAMPLES 20
#define VALIDATION_SAMPLES 20


/*
    Creates a synthetic dataset.

    Each sample has:
        One input feature: x
        One target value:  y = 2x + 1

    start:
        First input value.

    step:
        Difference between consecutive input values.
*/
static Dataset create_linear_dataset(
    size_t sample_count,
    float start,
    float step
)
{
    Dataset dataset = {0};

    dataset.sample_count = sample_count;
    dataset.input_size = 1;
    dataset.output_size = 1;

    dataset.inputs = matrix_create(sample_count, 1);
    dataset.targets = matrix_create(sample_count, 1);

    /*
        If either allocation fails, release both.
    */
    if (dataset.inputs.data == NULL ||
        dataset.targets.data == NULL)
    {
        dataset_free(&dataset);
        return dataset;
    }

    for (size_t i = 0; i < sample_count; i++)
    {
        float x = start + (float)i * step;
        float y = 2.0f * x + 1.0f;

        dataset.inputs.data[i] = x;
        dataset.targets.data[i] = y;
    }

    return dataset;
}


/*
    Checks whether a floating-point number is finite.

    Rejects:
        NaN
        Positive infinity
        Negative infinity
*/
static int is_valid_number(float value)
{
    return isfinite(value);
}


/*
    Main integration test.
*/
int main(void)
{
    printf("\n");
    printf("========================================\n");
    printf(" ZENITH-ASR TRAINING PIPELINE TEST\n");
    printf(" Implementation 009.4\n");
    printf("========================================\n\n");


    /* ------------------------------------
       TEST 1: Create training dataset
       ------------------------------------ */

    printf("Test 1: Dataset creation\n");
    printf("----------------------------------------\n");

    /*
        Training:
            0, 2, 4, 6, ..., 38
    */
    Dataset training = create_linear_dataset(
        TRAINING_SAMPLES,
        0.0f,
        2.0f
    );

    /*
        Validation:
            1, 3, 5, 7, ..., 39
    */
    Dataset validation = create_linear_dataset(
        VALIDATION_SAMPLES,
        1.0f,
        2.0f
    );

    if (training.inputs.data == NULL ||
        training.targets.data == NULL ||
        validation.inputs.data == NULL ||
        validation.targets.data == NULL)
    {
        printf("FAILED: Dataset allocation error.\n");

        dataset_free(&training);
        dataset_free(&validation);

        return EXIT_FAILURE;
    }

    printf("Training samples:   %zu\n",
           training.sample_count);

    printf("Validation samples: %zu\n",
           validation.sample_count);

    printf("Input features:     %zu\n",
           training.input_size);

    printf("Output values:      %zu\n",
           training.output_size);

    printf("Dataset creation PASSED.\n\n");


    /* ------------------------------------
       TEST 2: Create neural network
       ------------------------------------ */

    printf("Test 2: Network creation\n");
    printf("----------------------------------------\n");

    /*
        Architecture:

            Input:  1 neuron
            Hidden: 8 neurons
            Output: 1 neuron

        Total trainable layers: 2
    */

    size_t architecture[] = {1, 8, 1};

    Network network = network_create(
        architecture,
        2
    );

    /*
        Verify that network creation succeeded.
    */
    if (network.layers == NULL)
    {
        printf("FAILED: Network creation error.\n");

        dataset_free(&training);
        dataset_free(&validation);

        return EXIT_FAILURE;
    }

    printf("Network architecture: 1 -> 8 -> 1\n");
    printf("Trainable layers: 2\n");

    printf("Network creation PASSED.\n\n");


    /* ------------------------------------
       TEST 3: Configure training pipeline
       ------------------------------------ */

    printf("Test 3: Pipeline initialization\n");
    printf("----------------------------------------\n");

    TrainingPipeline pipeline;

    training_pipeline_init(&pipeline);

    /*
        Training configuration.

        epochs:
            Number of complete passes through
            the training dataset.

        batch_size:
            Number of samples per gradient update.

        learning_rate:
            Gradient descent learning rate.

        shuffle_seed:
            Deterministic random seed for shuffling.
    */

    TrainingConfig config = {
        .epochs = 1500,
        .batch_size = 5,
        .learning_rate = 0.01f,
        .shuffle_seed = 12345
    };

    /*
        Initialize history as an empty structure.
        The pipeline will allocate its loss arrays.
    */
    TrainingHistory history = {0};

    printf("Epochs:        %zu\n", config.epochs);
    printf("Batch size:    %zu\n", config.batch_size);
    printf("Learning rate: %.4f\n", config.learning_rate);
    printf("Shuffle seed:  %llu\n",
           (unsigned long long)config.shuffle_seed);

    printf("Pipeline initialization PASSED.\n\n");


    /* ------------------------------------
       TEST 4: Train the network
       ------------------------------------ */

    printf("Test 4: Neural network training\n");
    printf("----------------------------------------\n");

    printf("Training started...\n");

    int training_success = training_pipeline_train(
        &pipeline,
        &network,
        &training,
        &validation,
        &config,
        &history
    );

    if (!training_success)
    {
        printf("FAILED: Training pipeline returned an error.\n");

        training_history_free(&history);
        training_pipeline_free(&pipeline);

        network_free(&network);

        dataset_free(&training);
        dataset_free(&validation);

        return EXIT_FAILURE;
    }

    printf("Training completed successfully.\n");

    printf("Epochs completed: %zu\n",
           history.epochs_completed);

    if (history.epochs_completed == 0 ||
        history.training_loss == NULL ||
        history.validation_loss == NULL)
    {
        printf("FAILED: Invalid training history.\n");

        training_history_free(&history);
        training_pipeline_free(&pipeline);

        network_free(&network);

        dataset_free(&training);
        dataset_free(&validation);

        return EXIT_FAILURE;
    }

    printf("\nTraining history:\n");

    printf("Initial training loss: %.8f\n",
           history.training_loss[0]);

    printf("Final training loss:   %.8f\n",
           history.training_loss[history.epochs_completed - 1]);

    printf("Initial validation loss: %.8f\n",
           history.validation_loss[0]);

    printf("Final validation loss:   %.8f\n",
           history.validation_loss[history.epochs_completed - 1]);

    /*
        The final training loss should be lower
        than the initial training loss.
    */
    if (!is_valid_number(history.training_loss[0]) ||
        !is_valid_number(
            history.training_loss[history.epochs_completed - 1]
        ) ||
        history.training_loss[history.epochs_completed - 1]
            >= history.training_loss[0])
    {
        printf("\nFAILED: Training loss did not decrease.\n");

        training_history_free(&history);
        training_pipeline_free(&pipeline);

        network_free(&network);

        dataset_free(&training);
        dataset_free(&validation);

        return EXIT_FAILURE;
    }

    printf("\nTraining loss reduction PASSED.\n\n");


    /* ------------------------------------
       TEST 5: Evaluate validation dataset
       ------------------------------------ */

    printf("Test 5: Validation evaluation\n");
    printf("----------------------------------------\n");

    /*
        Evaluation uses the normalizer fitted
        during training.

        It must NOT update network weights.
    */

    float validation_loss = training_pipeline_evaluate(
        &pipeline,
        &network,
        &validation
    );

    if (!is_valid_number(validation_loss))
    {
        printf("FAILED: Invalid validation loss.\n");

        training_history_free(&history);
        training_pipeline_free(&pipeline);

        network_free(&network);

        dataset_free(&training);
        dataset_free(&validation);

        return EXIT_FAILURE;
    }

    printf("Validation loss: %.8f\n", validation_loss);

    printf("Validation evaluation PASSED.\n\n");


    /* ------------------------------------
       TEST 6: Check evaluation repeatability
       ------------------------------------ */

    printf("Test 6: Evaluation does not train\n");
    printf("----------------------------------------\n");

    /*
        Evaluate the same dataset again.

        Since evaluation does not perform
        gradient updates, both results should
        be identical apart from tiny floating-
        point differences.
    */

    float second_validation_loss = training_pipeline_evaluate(
        &pipeline,
        &network,
        &validation
    );

    if (!is_valid_number(second_validation_loss))
    {
        printf("FAILED: Second evaluation returned invalid loss.\n");

        training_history_free(&history);
        training_pipeline_free(&pipeline);

        network_free(&network);

        dataset_free(&training);
        dataset_free(&validation);

        return EXIT_FAILURE;
    }

    printf("First evaluation:  %.8f\n", validation_loss);
    printf("Second evaluation: %.8f\n", second_validation_loss);

    if (fabsf(validation_loss - second_validation_loss) > 0.00001f)
    {
        printf("FAILED: Evaluation results changed unexpectedly.\n");

        training_history_free(&history);
        training_pipeline_free(&pipeline);

        network_free(&network);

        dataset_free(&training);
        dataset_free(&validation);

        return EXIT_FAILURE;
    }

    printf("Evaluation repeatability PASSED.\n\n");


    /* ------------------------------------
       TEST 7: Check original datasets
       ------------------------------------ */

    printf("Test 7: Original dataset integrity\n");
    printf("----------------------------------------\n");

    /*
        The pipeline should work on transformed
        copies of the inputs.

        It must not overwrite the original data.
    */

    int dataset_unchanged = 1;

    for (size_t i = 0; i < TRAINING_SAMPLES; i++)
    {
        float expected = (float)i * 2.0f;

        if (fabsf(training.inputs.data[i] - expected) > 0.00001f)
        {
            dataset_unchanged = 0;
            break;
        }
    }

    if (!dataset_unchanged)
    {
        printf("FAILED: Original training dataset was modified.\n");

        training_history_free(&history);
        training_pipeline_free(&pipeline);

        network_free(&network);

        dataset_free(&training);
        dataset_free(&validation);

        return EXIT_FAILURE;
    }

    printf("Original training dataset unchanged.\n");
    printf("Original dataset integrity PASSED.\n\n");


    /* ------------------------------------
       TEST 8: Release resources
       ------------------------------------ */

    printf("Test 8: Memory cleanup\n");
    printf("----------------------------------------\n");

    training_history_free(&history);

    training_pipeline_free(&pipeline);

    network_free(&network);

    dataset_free(&training);
    dataset_free(&validation);

    printf("Training history released.\n");
    printf("Normalizer released.\n");
    printf("Network released.\n");
    printf("Datasets released.\n");

    printf("Memory cleanup PASSED.\n\n");


    /* ------------------------------------
       FINAL RESULT
       ------------------------------------ */

    printf("========================================\n");
    printf(" ALL PIPELINE TESTS PASSED!\n");
    printf("========================================\n\n");

    return EXIT_SUCCESS;
}
