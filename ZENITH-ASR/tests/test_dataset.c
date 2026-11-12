
#include "za_dataset.h"

#include <math.h>
#include <stdio.h>

int main(void)
{
    Dataset dataset = dataset_load_csv(
        "data/linear.csv",
        2,      /* Two input features. */
        1,      /* One target value. */
        1       /* CSV contains a header. */
    );

    if (dataset.inputs.data == NULL ||
        dataset.targets.data == NULL)
    {
        printf("Dataset loading failed!\n");
        dataset_free(&dataset);
        return 1;
    }

    printf("Dataset Loading Results\n");
    printf("-----------------------\n");

    printf("Samples:       %zu\n", dataset.sample_count);
    printf("Input features: %zu\n", dataset.input_size);
    printf("Output values:  %zu\n", dataset.output_size);

    printf("\nLoaded samples:\n");

    for (size_t i = 0; i < dataset.sample_count; i++)
    {
        printf(
            "Sample %zu: X = [%.2f, %.2f], Y = [%.2f]\n",
            i + 1,
            dataset.inputs.data[i * dataset.input_size],
            dataset.inputs.data[i * dataset.input_size + 1],
            dataset.targets.data[i * dataset.output_size]
        );
    }

    /*
     * Verify dimensions and known values.
     */
    if (dataset.sample_count != 4 ||
        dataset.input_size != 2 ||
        dataset.output_size != 1)
    {
        printf("\nTEST FAILED: Incorrect dimensions.\n");
        dataset_free(&dataset);
        return 1;
    }

    if (fabsf(dataset.inputs.data[0] - 1.0f) > 0.0001f ||
        fabsf(dataset.inputs.data[1] - 2.0f) > 0.0001f ||
        fabsf(dataset.targets.data[0] - 5.0f) > 0.0001f)
    {
        printf("\nTEST FAILED: Incorrect values.\n");
        dataset_free(&dataset);
        return 1;
    }

    dataset_free(&dataset);

    printf("\nDATASET TEST PASSED!\n");
    printf("All allocated resources released.\n");

    return 0;
}
