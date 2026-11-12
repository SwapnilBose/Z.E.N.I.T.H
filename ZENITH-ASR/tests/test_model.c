
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "za_matrix.h"
#include "za_dataset.h"
#include "za_network.h"
#include "za_pipeline.h"
#include "za_model.h"

/*
 * Create a synthetic dataset for:
 *
 *     y = 2x + 1
 *
 * Each row contains one input and one target.
 */
static Dataset create_linear_dataset(size_t sample_count,
                                     float start,
                                     float step)
{
    Dataset dataset = {0};

    dataset.inputs = matrix_create(sample_count, 1);
    dataset.targets = matrix_create(sample_count, 1);

    if (!dataset.inputs.data || !dataset.targets.data)
    {
        dataset_free(&dataset);
        return (Dataset){0};
    }

    dataset.sample_count = sample_count;
    dataset.input_size = 1;
    dataset.output_size = 1;

    for (size_t i = 0; i < sample_count; i++)
    {
        float x = start + (float)i * step;

        dataset.inputs.data[i] = x;
        dataset.targets.data[i] = 2.0f * x + 1.0f;
    }

    return dataset;
}


/*
 * Copy a file byte by byte.
 */
static int copy_file(const char *source_path,
                     const char *destination_path)
{
    FILE *source = fopen(source_path, "rb");

    if (!source)
        return 0;

    FILE *destination = fopen(destination_path, "wb");

    if (!destination)
    {
        fclose(source);
        return 0;
    }

    int success = 1;
    int byte;

    while ((byte = fgetc(source)) != EOF)
    {
        if (fputc(byte, destination) == EOF)
        {
            success = 0;
            break;
        }
    }

    if (ferror(source))
        success = 0;

    if (fclose(source) != 0)
        success = 0;

    if (fclose(destination) != 0)
        success = 0;

    return success;
}


/*
 * Corrupt one byte immediately before the final
 * eight-byte checksum.
 *
 * The model loader should detect this corruption.
 */
static int corrupt_model_file(const char *filename)
{
    FILE *file = fopen(filename, "r+b");

    if (!file)
        return 0;

    if (fseek(file, -9L, SEEK_END) != 0)
    {
        fclose(file);
        return 0;
    }

    int byte = fgetc(file);

    if (byte == EOF)
    {
        fclose(file);
        return 0;
    }

    if (fseek(file, -9L, SEEK_END) != 0)
    {
        fclose(file);
        return 0;
    }

    if (fputc(byte ^ 0x01, file) == EOF)
    {
        fclose(file);
        return 0;
    }

    return fclose(file) == 0;
}


/*
 * Compare two prediction matrices.
 */
static int predictions_match(const Matrix *a,
                             const Matrix *b,
                             float tolerance)
{
    if (!a || !b)
        return 0;

    if (!a->data || !b->data)
        return 0;

    if (a->rows != b->rows || a->cols != b->cols)
        return 0;

    size_t count = a->rows * a->cols;

    for (size_t i = 0; i < count; i++)
    {
        if (!isfinite(a->data[i]) || !isfinite(b->data[i]))
            return 0;

        if (fabsf(a->data[i] - b->data[i]) > tolerance)
            return 0;
    }

    return 1;
}


int main(void)
{
    const char *model_path = "models/test_model.zam";
    const char *corrupt_path = "models/test_model_corrupt.zam";

    int success = 0;

    Dataset training = {0};
    Dataset validation = {0};

    Network network = {0};

    TrainingPipeline pipeline;
    training_pipeline_init(&pipeline);

    TrainingHistory history = {0};

    Matrix raw_input = {0};
    Matrix normalized_input = {0};
    Matrix prediction_before = {0};
    Matrix prediction_after = {0};

    ZAModel loaded_model = {0};

    /*
     * TEST 1:
     * Create training and validation datasets.
     */
    printf("\n[TEST 1] Creating datasets...\n");

    training = create_linear_dataset(40, -2.0f, 0.1f);
    validation = create_linear_dataset(10, -1.95f, 0.37f);

    if (!training.inputs.data || !validation.inputs.data)
    {
        printf("FAIL: Dataset creation failed.\n");
        goto cleanup;
    }

    printf("PASS: Training and validation datasets created.\n");


    /*
     * TEST 2:
     * Create a neural network with architecture:
     *
     *     1 input -> 8 hidden neurons -> 1 output
     */
    printf("\n[TEST 2] Creating neural network...\n");

    const size_t architecture[] = {1, 8, 1};

    network = network_create(architecture, 2);

    if (!network.layers || network.layer_count != 2)
    {
        printf("FAIL: Network creation failed.\n");
        goto cleanup;
    }

    printf("PASS: Network created.\n");


    /*
     * TEST 3:
     * Train the network using the training pipeline.
     */
    printf("\n[TEST 3] Training network...\n");

    TrainingConfig config =
    {
        .epochs = 1500,
        .batch_size = 5,
        .learning_rate = 0.01f,
        .shuffle_seed = 12345
    };

    if (!training_pipeline_train(
            &pipeline,
            &network,
            &training,
            &validation,
            &config,
            &history))
    {
        printf("FAIL: Training pipeline failed.\n");
        goto cleanup;
    }

    printf("PASS: Network trained.\n");

    printf("Initial training loss: %.8f\n",
           history.training_loss[0]);

    printf("Final training loss:   %.8f\n",
           history.training_loss[history.epochs_completed - 1]);


    /*
     * TEST 4:
     * Create raw inputs for prediction.
     *
     * These values have not been normalized.
     */
    printf("\n[TEST 4] Preparing prediction inputs...\n");

    raw_input = matrix_create(3, 1);

    if (!raw_input.data)
    {
        printf("FAIL: Input matrix allocation failed.\n");
        goto cleanup;
    }

    raw_input.data[0] = -1.25f;
    raw_input.data[1] = 0.50f;
    raw_input.data[2] = 2.25f;

    /*
     * Normalize inputs using the fitted normalizer.
     */
    if (!normalizer_transform(
            &pipeline.input_normalizer,
            &raw_input,
            &normalized_input))
    {
        printf("FAIL: Input normalization failed.\n");
        goto cleanup;
    }

    /*
     * Obtain predictions from the original network.
     */
    prediction_before = network_forward(
        &network,
        &normalized_input
    );

    if (!prediction_before.data)
    {
        printf("FAIL: Original prediction failed.\n");
        goto cleanup;
    }

    printf("PASS: Original predictions generated.\n");


    /*
     * TEST 5:
     * Save network and normalizer.
     */
    printf("\n[TEST 5] Saving model...\n");

    if (!model_save(
            model_path,
            &network,
            &pipeline.input_normalizer))
    {
        printf("FAIL: Model saving failed.\n");
        goto cleanup;
    }

    printf("PASS: Model saved to %s\n", model_path);


    /*
     * TEST 6:
     * Destroy the original network and normalizer.
     *
     * This ensures that the loaded model cannot rely
     * on the original in-memory parameters.
     */
    printf("\n[TEST 6] Destroying original model...\n");

    matrix_free(&normalized_input);
    network_free(&network);
    training_pipeline_free(&pipeline);

    printf("PASS: Original network and normalizer destroyed.\n");


    /*
     * TEST 7:
     * Load the saved model.
     */
    printf("\n[TEST 7] Loading model...\n");

    loaded_model = model_load(model_path);

    if (!model_is_valid(&loaded_model))
    {
        printf("FAIL: Loaded model is invalid.\n");
        goto cleanup;
    }

    printf("PASS: Model loaded successfully.\n");


    /*
     * TEST 8:
     * Generate predictions using the loaded model.
     */
    printf("\n[TEST 8] Testing loaded model predictions...\n");

    prediction_after = model_predict(
        &loaded_model,
        &raw_input
    );

    if (!prediction_after.data)
    {
        printf("FAIL: Loaded model prediction failed.\n");
        goto cleanup;
    }

    printf("\nPrediction comparison:\n");

    for (size_t i = 0; i < prediction_before.rows; i++)
    {
        printf(
            "Input: %7.3f | Before: %10.6f | After: %10.6f\n",
            raw_input.data[i],
            prediction_before.data[i],
            prediction_after.data[i]
        );
    }

    if (!predictions_match(
            &prediction_before,
            &prediction_after,
            0.000001f))
    {
        printf("FAIL: Predictions changed after loading.\n");
        goto cleanup;
    }

    printf("PASS: Predictions preserved after loading.\n");


    /*
     * TEST 9:
     * Corrupt a copy of the model file.
     *
     * The checksum should detect the modification.
     */
    printf("\n[TEST 9] Testing corruption detection...\n");

    if (!copy_file(model_path, corrupt_path))
    {
        printf("FAIL: Could not copy model file.\n");
        goto cleanup;
    }

    if (!corrupt_model_file(corrupt_path))
    {
        printf("FAIL: Could not corrupt test file.\n");
        goto cleanup;
    }

    ZAModel corrupted_model = model_load(corrupt_path);

    if (model_is_valid(&corrupted_model))
    {
        printf("FAIL: Corrupted model was accepted.\n");
        model_free(&corrupted_model);
        goto cleanup;
    }

    model_free(&corrupted_model);

    printf("PASS: Corrupted model correctly rejected.\n");


    /*
     * All tests passed.
     */
    printf("\n====================================\n");
    printf(" ALL MODEL PERSISTENCE TESTS PASSED\n");
    printf("====================================\n");

    success = 1;


cleanup:

    /*
     * Release all resources.
     *
     * These cleanup functions should also be safe
     * when called on empty objects.
     */
    matrix_free(&raw_input);
    matrix_free(&normalized_input);
    matrix_free(&prediction_before);
    matrix_free(&prediction_after);

    dataset_free(&training);
    dataset_free(&validation);

    network_free(&network);
    training_pipeline_free(&pipeline);
    training_history_free(&history);

    model_free(&loaded_model);

    remove(corrupt_path);

    if (!success)
        printf("\nMODEL PERSISTENCE TEST FAILED.\n");

    return success ? 0 : 1;
}
