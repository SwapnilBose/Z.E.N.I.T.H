#!/usr/bin/env bash
set -euo pipefail

if [[ "${MSYSTEM:-}" != "UCRT64" ]]; then
  echo "Open the MSYS2 UCRT64 terminal, then run: bash build-ucrt64.sh"
  exit 1
fi

command -v gcc >/dev/null 2>&1 || { echo "gcc is missing."; exit 1; }
command -v pkg-config >/dev/null 2>&1 || { echo "pkg-config is missing."; exit 1; }
pkg-config --exists portaudio-2.0 || {
  echo "PortAudio development package not found. Install: pacman -S mingw-w64-ucrt-x86_64-portaudio mingw-w64-ucrt-x86_64-pkgconf"
  exit 1
}

mkdir -p build
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic \
  -Iinclude $(pkg-config --cflags portaudio-2.0) \
  src/main.c src/audio.c src/vosk_dynamic.c src/json.c src/tts.c src/transcript.c \
  -o build/zenith.exe \
  $(pkg-config --libs portaudio-2.0) \
  -lwinmm -lole32 -luuid -lsetupapi

echo "Built: build/zenith.exe"
echo "Run from project root: ./build/zenith.exe"
