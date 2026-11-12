
#include "za_model.h"

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>

/*
 * ZENITH MODEL FORMAT
 *
 * Magic:          8 bytes
 * Version:        uint32_t
 * Layer count:    uint64_t
 * Layer metadata: input_size, output_size
 * Layer data:     weights, biases
 * Normalizer:     feature_count, means, scales
 * Checksum:       uint64_t
 *
 * All integer fields use fixed-width integer types.
 * Float values use the host's native float representation.
 */

#define MODEL_MAGIC "ZENITHM1"
#define MODEL_MAGIC_SIZE 8

#define MODEL_FORMAT_VERSION 1

/*
 * Defensive limits.
 *
 * These limits prevent corrupted files from requesting
 * unreasonable amounts of memory during loading.
 */
#define MODEL_MAX_LAYERS 128
#define MODEL_MAX_DIMENSION 1000000
#define MODEL_MAX_PARAMETERS 10000000ULL

/*
 * FNV-1a 64-bit checksum constants.
 */
#define FNV_OFFSET_BASIS 14695981039346656037ULL
#define FNV_PRIME 1099511628211ULL


/* =========================================================
   CHECKSUM
   ========================================================= */

/*
 * Update the checksum one byte at a time.
 */
static uint64_t checksum_update(
    uint64_t hash,
    const void *data,
    size_t size
)
{
    const unsigned char *bytes = data;

    for (size_t i = 0; i < size; i++)
    {
        hash ^= bytes[i];
        hash *= FNV_PRIME;
    }

    return hash;
}


/* =========================================================
   FILE I/O HELPERS
   ========================================================= */

/*
 * Write bytes and include them in the checksum.
 */
static int write_hashed(
    FILE *file,
    const void *data,
    size_t size,
    uint64_t *hash
)
{
    if (fwrite(data, 1, size, file) != size)
        return 0;

    *hash = checksum_update(*hash, data, size);

    return 1;
}


/*
 * Read bytes and include them in the checksum.
 */
static int read_hashed(
    FILE *file,
    void *data,
    size_t size,
    uint64_t *hash
)
{
    if (fread(data, 1, size, file) != size)
        return 0;

    *hash = checksum_update(*hash, data, size);

    return 1;
}


/*
 * Write a uint64_t value.
 */
static int write_u64(
    FILE *file,
    uint64_t value,
    uint64_t *hash
)
{
    return write_hashed(
        file,
        &value,
        sizeof(value),
        hash
    );
}


/*
 * Read a uint64_t value.
 */
static int read_u64(
    FILE *file,
    uint64_t *value,
    uint64_t *hash
)
{
    return read_hashed(
        file,
        value,
        sizeof(*value),
        hash
    );
}


/*
 * Write a uint32_t value.
 */
static int write_u32(
    FILE *file,
    uint32_t value,
    uint64_t *hash
)
{
    return write_hashed(
        file,
        &value,
        sizeof(value),
        hash
    );
}


/*
 * Read a uint32_t value.
 */
static int read_u32(
    FILE *file,
    uint32_t *value,
    uint64_t *hash
)
{
    return read_hashed(
        file,
        value,
        sizeof(*value),
        hash
    );
}


/* =========================================================
   VALIDATION HELPERS
   ========================================================= */

/*
 * Check whether a matrix has valid dimensions and storage.
 */
static int matrix_is_valid(const Matrix *matrix)
{
    if (matrix == NULL)
        return 0;

    if (matrix->rows == 0 || matrix->cols == 0)
        return 0;

    if (matrix->data == NULL)
        return 0;

    if (matrix->rows > SIZE_MAX / matrix->cols)
        return 0;

    size_t count = matrix->rows * matrix->cols;

    for (size_t i = 0; i < count; i++)
    {
        if (!isfinite(matrix->data[i]))
            return 0;
    }

    return 1;
}


/*
 * Validate a network before saving.
 *
 * Checks:
 * - Network exists.
 * - Every layer has valid dimensions.
 * - Weight and bias matrices match layer dimensions.
 * - Adjacent layers connect correctly.
 * - Parameters are finite.
 */
static int network_is_valid(const Network *network)
{
    if (network == NULL)
        return 0;

    if (network->layers == NULL)
        return 0;

    if (network->layer_count == 0 ||
        network->layer_count > MODEL_MAX_LAYERS)
        return 0;

    uint64_t total_parameters = 0;

    for (size_t i = 0; i < network->layer_count; i++)
    {
        const DenseLayer *layer = &network->layers[i];

        if (layer->input_size == 0 ||
            layer->output_size == 0)
            return 0;

        if (layer->input_size > MODEL_MAX_DIMENSION ||
            layer->output_size > MODEL_MAX_DIMENSION)
            return 0;

        if (layer->weights.rows != layer->input_size ||
            layer->weights.cols != layer->output_size)
            return 0;

        if (layer->biases.rows != 1 ||
            layer->biases.cols != layer->output_size)
            return 0;

        if (!matrix_is_valid(&layer->weights) ||
            !matrix_is_valid(&layer->biases))
            return 0;

        if (i > 0 &&
            network->layers[i - 1].output_size != layer->input_size)
            return 0;

        uint64_t weight_count =
            (uint64_t)layer->input_size *
            (uint64_t)layer->output_size;

        uint64_t bias_count = (uint64_t)layer->output_size;

        if (weight_count > MODEL_MAX_PARAMETERS ||
            bias_count > MODEL_MAX_PARAMETERS - weight_count)
            return 0;

        uint64_t layer_parameters = weight_count + bias_count;

        if (total_parameters >
            MODEL_MAX_PARAMETERS - layer_parameters)
            return 0;

        total_parameters += layer_parameters;
    }

    return 1;
}


/*
 * Validate a normalizer.
 */
static int normalizer_is_valid(const Normalizer *normalizer)
{
    if (normalizer == NULL)
        return 0;

    if (normalizer->feature_count == 0)
        return 0;

    if (normalizer->means == NULL ||
        normalizer->scales == NULL)
        return 0;

    if (normalizer->feature_count > MODEL_MAX_DIMENSION)
        return 0;

    for (size_t i = 0; i < normalizer->feature_count; i++)
    {
        if (!isfinite(normalizer->means[i]))
            return 0;

        if (!isfinite(normalizer->scales[i]))
            return 0;

        if (normalizer->scales[i] <= 0.0f)
            return 0;
    }

    return 1;
}


/* =========================================================
   MODEL VALIDATION
   ========================================================= */

int model_is_valid(const ZAModel *model)
{
    if (model == NULL)
        return 0;

    if (!network_is_valid(&model->network))
        return 0;

    if (!normalizer_is_valid(&model->normalizer))
        return 0;

    /*
     * The normalizer must match the network input size.
     */
    if (model->normalizer.feature_count !=
        model->network.layers[0].input_size)
        return 0;

    return 1;
}


/* =========================================================
   MODEL SAVING
   ========================================================= */

int model_save(
    const char *filename,
    const Network *network,
    const Normalizer *normalizer
)
{
    /*
     * Validate all arguments before touching the file.
     */
    if (filename == NULL || filename[0] == '\0')
        return 0;

    if (!network_is_valid(network))
        return 0;

    if (!normalizer_is_valid(normalizer))
        return 0;

    if (normalizer->feature_count !=
        network->layers[0].input_size)
        return 0;

    /*
     * This version requires 32-bit IEEE-754 floats.
     */
    if (sizeof(float) != 4 ||
        FLT_RADIX != 2 ||
        FLT_MANT_DIG != 24)
        return 0;

    FILE *file = fopen(filename, "wb");

    if (file == NULL)
        return 0;

    uint64_t hash = FNV_OFFSET_BASIS;

    int success = 1;

    /*
     * 1. Write magic signature.
     */
    const unsigned char magic[MODEL_MAGIC_SIZE] =
    {
        'Z', 'E', 'N', 'I', 'T', 'H', 'M', '1'
    };

    if (!write_hashed(
            file,
            magic,
            MODEL_MAGIC_SIZE,
            &hash))
    {
        success = 0;
    }

    /*
     * 2. Write format version.
     */
    if (success &&
        !write_u32(file, MODEL_FORMAT_VERSION, &hash))
    {
        success = 0;
    }

    /*
     * 3. Write number of trainable layers.
     */
    if (success &&
        !write_u64(
            file,
            (uint64_t)network->layer_count,
            &hash))
    {
        success = 0;
    }

    
    /*
     * 4. Write ALL layer dimensions first.
     *
     * This allows the loader to reconstruct the
     * complete architecture before allocating
     * the network.
     */
    for (size_t i = 0;
         success && i < network->layer_count;
         i++)
    {
        const DenseLayer *layer = &network->layers[i];

        if (!write_u64(
                file,
                (uint64_t)layer->input_size,
                &hash))
        {
            success = 0;
            break;
        }

        if (!write_u64(
                file,
                (uint64_t)layer->output_size,
                &hash))
        {
            success = 0;
            break;
        }
    }

    /*
     * 5. Write ALL weights and biases.
     *
     * The order must match the loader:
     * Layer 1 weights -> Layer 1 biases
     * Layer 2 weights -> Layer 2 biases
     * ...
     */
    for (size_t i = 0;
         success && i < network->layer_count;
         i++)
    {
        const DenseLayer *layer = &network->layers[i];

        size_t weight_count =
            layer->weights.rows * layer->weights.cols;

        size_t bias_count =
            layer->biases.rows * layer->biases.cols;

        if (!write_hashed(
                file,
                layer->weights.data,
                weight_count * sizeof(float),
                &hash))
        {
            success = 0;
            break;
        }

        if (!write_hashed(
                file,
                layer->biases.data,
                bias_count * sizeof(float),
                &hash))
        {
            success = 0;
            break;
        }
    }

    /*
     * Continue with the normalizer feature count,
     * means, scales and checksum as before.
     */


    /*
     * 5. Write normalizer feature count.
     */
    if (success &&
        !write_u64(
            file,
            (uint64_t)normalizer->feature_count,
            &hash))
    {
        success = 0;
    }

    /*
     * 6. Write feature means.
     */
    if (success &&
        !write_hashed(
            file,
            normalizer->means,
            normalizer->feature_count * sizeof(float),
            &hash))
    {
        success = 0;
    }

    /*
     * 7. Write feature scales.
     */
    if (success &&
        !write_hashed(
            file,
            normalizer->scales,
            normalizer->feature_count * sizeof(float),
            &hash))
    {
        success = 0;
    }

    /*
     * 8. Write final checksum.
     *
     * The checksum itself is NOT included
     * in the checksum calculation.
     */
    if (success &&
        fwrite(&hash, sizeof(hash), 1, file) != 1)
    {
        success = 0;
    }

    /*
     * Flush buffered data before closing.
     */
    if (success && fflush(file) != 0)
        success = 0;

    if (fclose(file) != 0)
        success = 0;

    return success;
}


/* =========================================================
   MODEL LOADING
   ========================================================= */

ZAModel model_load(const char *filename)
{
    ZAModel model = {0};

    if (filename == NULL || filename[0] == '\0')
        return model;

    if (sizeof(float) != 4 ||
        FLT_RADIX != 2 ||
        FLT_MANT_DIG != 24)
        return model;

    FILE *file = fopen(filename, "rb");

    if (file == NULL)
        return model;

    uint64_t hash = FNV_OFFSET_BASIS;

    char magic[MODEL_MAGIC_SIZE];

    uint32_t version = 0;
    uint64_t layer_count_u64 = 0;

    size_t *sizes = NULL;

    int success = 1;

    /*
     * 1. Read and verify magic signature.
     */
    if (!read_hashed(
            file,
            magic,
            MODEL_MAGIC_SIZE,
            &hash))
    {
        success = 0;
    }

    if (success &&
        memcmp(magic, MODEL_MAGIC, MODEL_MAGIC_SIZE) != 0)
    {
        success = 0;
    }

    /*
     * 2. Read format version.
     */
    if (success &&
        !read_u32(file, &version, &hash))
    {
        success = 0;
    }

    if (success && version != MODEL_FORMAT_VERSION)
    {
        success = 0;
    }

    /*
     * 3. Read layer count.
     */
    if (success &&
        !read_u64(file, &layer_count_u64, &hash))
    {
        success = 0;
    }

    if (success &&
        (layer_count_u64 == 0 ||
         layer_count_u64 > MODEL_MAX_LAYERS))
    {
        success = 0;
    }

    if (!success)
        goto cleanup;

    size_t layer_count = (size_t)layer_count_u64;

    /*
     * Architecture has layer_count + 1 dimensions.
     */
    if (layer_count == SIZE_MAX ||
        layer_count + 1 > SIZE_MAX / sizeof(size_t))
    {
        success = 0;
        goto cleanup;
    }

    sizes = calloc(layer_count + 1, sizeof(size_t));

    if (sizes == NULL)
    {
        success = 0;
        goto cleanup;
    }

    /*
     * 4. Read layer dimensions.
     */
    uint64_t total_parameters = 0;

    for (size_t i = 0; i < layer_count; i++)
    {
        uint64_t input_size = 0;
        uint64_t output_size = 0;

        if (!read_u64(file, &input_size, &hash) ||
            !read_u64(file, &output_size, &hash))
        {
            success = 0;
            break;
        }

        if (input_size == 0 ||
            output_size == 0 ||
            input_size > MODEL_MAX_DIMENSION ||
            output_size > MODEL_MAX_DIMENSION ||
            input_size > SIZE_MAX ||
            output_size > SIZE_MAX)
        {
            success = 0;
            break;
        }

        if (i == 0)
            sizes[0] = (size_t)input_size;

        else if (sizes[i] != (size_t)input_size)
        {
            success = 0;
            break;
        }

        sizes[i + 1] = (size_t)output_size;

        uint64_t weight_count = input_size * output_size;
        uint64_t bias_count = output_size;

        if (weight_count > MODEL_MAX_PARAMETERS ||
            bias_count > MODEL_MAX_PARAMETERS - weight_count)
        {
            success = 0;
            break;
        }

        uint64_t layer_parameters = weight_count + bias_count;

        if (total_parameters >
            MODEL_MAX_PARAMETERS - layer_parameters)
        {
            success = 0;
            break;
        }

        total_parameters += layer_parameters;
    }

    if (!success)
        goto cleanup;

    /*
     * 5. Reconstruct the network architecture.
     *
     * network_create initializes parameters.
     * We overwrite them with the saved values below.
     */
    model.network = network_create(sizes, layer_count);

    if (model.network.layers == NULL)
    {
        success = 0;
        goto cleanup;
    }

    /*
     * 6. Read weights and biases.
     */
    for (size_t i = 0; i < layer_count; i++)
    {
        DenseLayer *layer = &model.network.layers[i];

        size_t weight_count =
            layer->weights.rows * layer->weights.cols;

        size_t bias_count =
            layer->biases.rows * layer->biases.cols;

        if (!read_hashed(
                file,
                layer->weights.data,
                weight_count * sizeof(float),
                &hash))
        {
            success = 0;
            break;
        }

        if (!read_hashed(
                file,
                layer->biases.data,
                bias_count * sizeof(float),
                &hash))
        {
            success = 0;
            break;
        }

        /*
         * Reject NaN and infinite parameters.
         */
        if (!matrix_is_valid(&layer->weights) ||
            !matrix_is_valid(&layer->biases))
        {
            success = 0;
            break;
        }
    }

    if (!success)
        goto cleanup;

    /*
     * 7. Read normalizer feature count.
     */
    uint64_t feature_count_u64 = 0;

    if (!read_u64(file, &feature_count_u64, &hash))
    {
        success = 0;
        goto cleanup;
    }

    if (feature_count_u64 == 0 ||
        feature_count_u64 > MODEL_MAX_DIMENSION ||
        feature_count_u64 > SIZE_MAX)
    {
        success = 0;
        goto cleanup;
    }

    size_t feature_count = (size_t)feature_count_u64;

    if (feature_count != sizes[0])
    {
        success = 0;
        goto cleanup;
    }

    /*
     * 8. Allocate normalizer arrays.
     */
    model.normalizer.means =
        malloc(feature_count * sizeof(float));

    model.normalizer.scales =
        malloc(feature_count * sizeof(float));

    if (model.normalizer.means == NULL ||
        model.normalizer.scales == NULL)
    {
        success = 0;
        goto cleanup;
    }

    model.normalizer.feature_count = feature_count;

    /*
     * 9. Read means and scales.
     */
    if (!read_hashed(
            file,
            model.normalizer.means,
            feature_count * sizeof(float),
            &hash))
    {
        success = 0;
        goto cleanup;
    }

    if (!read_hashed(
            file,
            model.normalizer.scales,
            feature_count * sizeof(float),
            &hash))
    {
        success = 0;
        goto cleanup;
    }

    /*
     * Validate the restored normalizer.
     */
    if (!normalizer_is_valid(&model.normalizer))
    {
        success = 0;
        goto cleanup;
    }

    /*
     * 10. Read the stored checksum.
     */
    uint64_t stored_hash = 0;

    if (fread(&stored_hash, sizeof(stored_hash), 1, file) != 1)
    {
        success = 0;
        goto cleanup;
    }

    /*
     * Compare calculated and stored checksums.
     */
    if (stored_hash != hash)
    {
        success = 0;
        goto cleanup;
    }

    /*
     * Reject trailing data.
     */
    if (fgetc(file) != EOF || ferror(file))
    {
        success = 0;
        goto cleanup;
    }

    /*
     * Final structural validation.
     */
    if (!model_is_valid(&model))
    {
        success = 0;
        goto cleanup;
    }

cleanup:

    free(sizes);

    if (fclose(file) != 0)
        success = 0;

    if (!success)
    {
        model_free(&model);
        return (ZAModel){0};
    }

    return model;
}


/* =========================================================
   MODEL PREDICTION
   ========================================================= */

Matrix model_predict(
    const ZAModel *model,
    const Matrix *input
)
{
    Matrix empty = {0};

    /*
     * Validate model.
     */
    if (!model_is_valid(model))
        return empty;

    /*
     * Validate input matrix.
     */
    if (!matrix_is_valid(input))
        return empty;

    /*
     * Input feature count must match the normalizer.
     */
    if (input->cols != model->normalizer.feature_count)
        return empty;

    /*
     * Normalize raw input using saved training statistics.
     */
    Matrix normalized = {0};

    if (!normalizer_transform(
            &model->normalizer,
            input,
            &normalized))
    {
        return empty;
    }

    /*
     * Perform neural network inference.
     */
    Matrix prediction = network_forward(
        &model->network,
        &normalized
    );

    /*
     * Release temporary normalized input.
     */
    matrix_free(&normalized);

    return prediction;
}


/* =========================================================
   MODEL CLEANUP
   ========================================================= */

void model_free(ZAModel *model)
{
    if (model == NULL)
        return;

    network_free(&model->network);

    normalizer_free(&model->normalizer);

    /*
     * Reset the structure.
     */
    *model = (ZAModel){0};
}
