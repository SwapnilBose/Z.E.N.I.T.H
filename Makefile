CC = gcc
PKG_CONFIG ?= pkg-config

CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Wpedantic
CPPFLAGS += -Iinclude
PA_CFLAGS := $(shell $(PKG_CONFIG) --cflags portaudio-2.0 2>/dev/null)
PA_LIBS := $(shell $(PKG_CONFIG) --libs portaudio-2.0 2>/dev/null)

SOURCES := src/main.c src/audio.c src/vosk_dynamic.c src/json.c src/tts.c src/transcript.c
TARGET := build/zenith.exe

.PHONY: all clean check

all: check $(TARGET)

check:
	@command -v $(CC) >/dev/null 2>&1 || { echo "gcc not found. Open the MSYS2 UCRT64 terminal and install the UCRT64 GCC package."; exit 1; }
	@test -n "$(PA_LIBS)" || { echo "PortAudio pkg-config metadata not found. Install mingw-w64-ucrt-x86_64-portaudio and mingw-w64-ucrt-x86_64-pkgconf."; exit 1; }

$(TARGET): $(SOURCES) $(wildcard include/*.h)
	@mkdir -p build
	$(CC) $(CFLAGS) $(CPPFLAGS) $(PA_CFLAGS) $(SOURCES) -o $(TARGET) $(PA_LIBS) -lwinmm -lole32 -luuid -lsetupapi

clean:
	rm -rf build
