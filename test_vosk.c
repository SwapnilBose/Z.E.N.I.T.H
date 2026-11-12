
#include <windows.h>
#include <stdio.h>

int main(void)
{
    SetDllDirectoryA("bin");

    printf("Testing Vosk DLL loading...\n");

    HMODULE dll = LoadLibraryExA(
        "bin/libvosk.dll",
        NULL,
        LOAD_WITH_ALTERED_SEARCH_PATH
    );

    if (dll == NULL)
    {
        DWORD error = GetLastError();

        printf("FAILED: Windows error %lu\n", error);

        char *message = NULL;

        FormatMessageA(
            FORMAT_MESSAGE_ALLOCATE_BUFFER |
            FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL,
            error,
            0,
            (LPSTR)&message,
            0,
            NULL
        );

        if (message)
        {
            printf("Details: %s\n", message);
            LocalFree(message);
        }

        return 1;
    }

    printf("SUCCESS: Vosk DLL loaded!\n");

    FARPROC function = GetProcAddress(dll, "vosk_model_new");

    if (function)
        printf("SUCCESS: vosk_model_new export found!\n");
    else
        printf("FAILED: vosk_model_new export missing.\n");

    FreeLibrary(dll);

    return 0;
}
