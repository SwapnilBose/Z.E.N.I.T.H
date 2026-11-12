# Z.E.N.I.T.H.

**Z.E.N.I.T.H. — Zen Engine for Natural Input and Transcription Handling**

An offline Windows voice input and spoken-replay module written in C.

- **Speech input:** PortAudio captures microphone audio.
- **Speech recognition:** Vosk converts audio to text locally.
- **Transcript:** recognized utterances are appended to `data/transcripts.txt`.
- **Spoken replay:** eSpeak NG reads the recognized text back.
- **No Python, cloud API, or internet connection is needed at runtime.**

This ZIP contains the C source, headers, Makefile, run script, setup instructions, and folder placeholders. It does **not** contain the external PortAudio, Vosk, eSpeak NG binaries or the Vosk model. Those are separate downloads/installs and are often several MB to hundreds of MB.

---

## 1. What you need to install

### A. MSYS2 + MinGW-w64 UCRT64

1. Install MSYS2: <https://www.msys2.org/>
2. Open **MSYS2 UCRT64** from the Start menu (not the plain MSYS terminal).
3. Update packages:

```bash
pacman -Syu
```

If the terminal closes after an update, reopen **MSYS2 UCRT64** and run the update command again if requested.

4. Install the compiler, Make, pkg-config and PortAudio:

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make mingw-w64-ucrt-x86_64-pkgconf mingw-w64-ucrt-x86_64-portaudio
```

This installs a compatible prebuilt PortAudio development package (header, import library and runtime DLL). It is the easiest route. Your `portaudio.tgz` archive is optional; see **Building your PortAudio archive** below.

### B. Vosk Windows x64 runtime

Download the Windows x64 Vosk API package from the official releases:

<https://github.com/alphacep/vosk-api/releases>

The commonly used Windows release asset is `vosk-win64-0.3.45.zip`.

1. Extract the archive.
2. Copy `libvosk.dll` into this project's `bin/` folder.
3. Copy any companion runtime DLLs supplied beside `libvosk.dll` into `bin/` too. Keep the DLLs together.

The program loads `bin/libvosk.dll` dynamically, so a Vosk `.a` or `.lib` import library is not required.

### C. Vosk small English model

You said you already downloaded the small Vosk model.

1. Extract the model ZIP.
2. Copy the extracted model folder into this project's `models/` folder.
3. The expected default path is:

```text
models/vosk-model-small-en-us-0.15/
```

The model folder should contain items such as `am`, `conf`, `graph` and `ivector`. Do not leave it as a ZIP file.

If your extracted folder has a different name, either rename it to the default above or launch with `--model "models\your-model-folder"`.

Official model list: <https://alphacephei.com/vosk/models>

### D. eSpeak NG

You said you already downloaded eSpeak NG.

1. Run its Windows x64 installer.
2. Confirm the executable is available in PATH. Open a new Command Prompt and run:

```bat
espeak-ng --version
```

If Windows says the command is not recognized, add the installed eSpeak NG directory to PATH. Common installer location: `C:\Program Files\eSpeak NG`.

Official releases: <https://github.com/espeak-ng/espeak-ng/releases>

---

## 2. Build Z.E.N.I.T.H.

Extract this project ZIP to a simple path, for example:

```text
C:\dev\ZENITH\
```

Avoid a path containing non-ASCII characters for the first build, because the Vosk C API accepts narrow character paths.

Open **MSYS2 UCRT64**, change directory to the project folder (MSYS path syntax):

```bash
cd /c/dev/ZENITH
```

Build:

```bash
bash build-ucrt64.sh
```

Or run:

```bash
make
```

The executable will be created at:

```text
build/zenith.exe
```

### If Make cannot find PortAudio

Check:

```bash
pkg-config --modversion portaudio-2.0
```

If that fails, confirm you opened the **UCRT64** terminal and installed `mingw-w64-ucrt-x86_64-portaudio` plus `mingw-w64-ucrt-x86_64-pkgconf`.

---

## 3. Run it

From the project root, in MSYS2 UCRT64:

```bash
./build/zenith.exe
```

Or double-click `run.bat` after building. `run.bat` assumes MSYS2 is installed in `C:\msys64`; edit the path in that file if yours is elsewhere.

When ready:

- Speak naturally.
- Vosk displays changing partial text while you speak.
- Vosk detects the end of an utterance after a pause.
- The final recognized text appears as `YOU:`.
- eSpeak NG speaks the recognized text back as `ZENITH (spoken replay)`.
- The transcript is saved to `data/transcripts.txt`.
- Press **Q** or **ESC** to exit.

The replay is synthesized speech, not a replay of the original microphone waveform.

---

## 4. Command-line options

```text
build/zenith.exe --help
```

Examples:

Disable spoken replay but keep transcription:

```bash
./build/zenith.exe --no-replay
```

Specify a differently named model folder:

```bash
./build/zenith.exe --model "models/vosk-model-small-en-us-0.15"
```

Change the eSpeak voice and speed:

```bash
./build/zenith.exe --voice en-us --rate 175
```

The eSpeak rate is in words per minute; the accepted range is 80–450.

---

## 5. You have `portaudio.tgz`: what to do with it

You do not need to compile it if the MSYS2 PortAudio package installed successfully. If you specifically want to build the source archive you downloaded:

1. Open MSYS2 UCRT64.
2. Install CMake and Ninja:

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja
```

3. Extract the archive. Replace the archive name below if yours is named differently:

```bash
mkdir -p /c/dev/portaudio-src
cd /c/dev/portaudio-src
tar -xzf /c/Users/YOUR_WINDOWS_USER/Downloads/portaudio.tgz
```

4. Enter the extracted PortAudio source directory (the folder containing `CMakeLists.txt`) and build:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

For Z.E.N.I.T.H., the prebuilt MSYS2 package is simpler because it supplies the PortAudio header, import library, pkg-config metadata and runtime DLL together. Avoid mixing libraries built by different compiler environments (UCRT64, MINGW64, Visual Studio, etc.). PortAudio's official MinGW build guide: <https://files.portaudio.com/docs/v19-doxydocs/compile_windows_mingw-w64.html>.

---

## 6. Troubleshooting

| Symptom | Check |
|---|---|
| `libvosk.dll` cannot load | Confirm `bin/libvosk.dll` exists and copy all companion DLLs from the Vosk archive into `bin/`. |
| Model not found | Confirm the extracted model directory is under `models/` and contains `am/`, `conf/`, etc. |
| `portaudio-2.0` not found | Open MSYS2 UCRT64 and install the PortAudio and pkgconf packages. |
| No default microphone | Choose a default input device in Windows Settings → System → Sound. |
| Microphone permission denied | Windows Settings → Privacy & security → Microphone; enable desktop app access. |
| 16 kHz input rejected | Z.E.N.I.T.H. automatically retries at 48 kHz. |
| eSpeak cannot launch | Run `espeak-ng --version` in a new terminal and check PATH. |
| Recognition is poor | Move closer to the mic, reduce background noise, and test another Vosk model. |
| Speaker audio is re-recognized | Capture pauses during eSpeak playback; use headphones if acoustic leakage continues. |

---

## 7. Source layout

```text
ZENITH/
├── README.md
├── Makefile
├── build-ucrt64.sh
├── run.bat
├── include/
│   ├── zenith_config.h
│   ├── zenith_audio.h
│   ├── zenith_vosk.h
│   ├── zenith_json.h
│   ├── zenith_tts.h
│   └── zenith_transcript.h
├── src/
│   ├── main.c
│   ├── audio.c
│   ├── vosk_dynamic.c
│   ├── json.c
│   ├── tts.c
│   └── transcript.c
├── models/       (put extracted Vosk model here)
├── bin/          (put Vosk DLLs here)
├── data/         (transcript log is created here)
└── docs/
    └── ARCHITECTURE.md
```

## 8. Known first-version limitations

- English recognition is configured by whichever model you install; the default folder name is for a small US English model.
- The app uses the Windows default microphone, not a device-selection menu.
- It uses Vosk endpoint detection to decide when an utterance is complete; it does not include a separate neural voice-activity detector.
- Speech is replayed with eSpeak NG's synthetic voice. Original voice recording playback is not implemented in this version.
- This is the standalone speech/transcription component. It does not yet parse mathematical language or call Zen-π's solver.

See `docs/ARCHITECTURE.md` before extending the modules.
