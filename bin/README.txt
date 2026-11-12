Copy the Vosk Windows x64 runtime DLLs into this folder.

At minimum, the primary library must be:
  bin/libvosk.dll

Copy any companion DLLs included in the same Vosk Windows package alongside it
(for example MinGW runtime DLLs, if present in the package). Do not copy the
Vosk model into this folder; the model belongs in models/.
