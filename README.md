# MOSS-Image-Toriigate-dataset-autocaptioner

Dataset auto-captioner for MOSS-Image, using the Minthy/ToriiGate-0.5 model (and Qwen3.8-27B as an alternate) for captioning. Manages local [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` instances with multi-GPU support, dynamic KV-cache sizing, and specialized captioning presets (MOSS-Image JSON schema, Booru/Danbooru tags).

## Requirements

- Ubuntu / Xubuntu 24.04 or 26.04 (or any Linux with a working desktop session)
- Python **3.12+**
- A CUDA (NVIDIA) or ROCm (AMD) GPU — the app also runs without one, but inference requires `llama-server` built for your platform
- `llama-server` available on your `PATH`, or set via **Tab 2 → Llama.cpp Args / binary path** (build llama.cpp with CUDA/ROCm support for GPU offloading)

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-user>/MOSS-Image-Toriigate-dataset-autocaptioner.git
   cd MOSS-Image-Toriigate-dataset-autocaptioner
   ```

2. **Create a virtual environment** (recommended — keeps system packages untouched)

   ```bash
   sudo apt update
   sudo apt install -y python3-venv python3-pip   # if not already installed

   python3 -m venv .venv
   source .venv/bin/activate
   ```

   > You will need to run `source .venv/bin/activate` in every new terminal before using the app. To leave the environment later, run `deactivate`.

3. **Install Python dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   On some minimal Xubuntu installs PySide6 needs these system libraries:

   ```bash
   sudo apt install -y libxcb-cursor0 libxkbcommon-x11-0 libegl1
   ```

4. **(Optional) Install NVIDIA driver monitoring libs** — handled automatically by `nvidia-ml-py` from `requirements.txt`. For AMD GPUs, make sure `rocm-smi` is on your `PATH`.

## Running

```bash
source .venv/bin/activate   # if not already active
./run.sh
```

or directly:

```bash
python -m app.main
```

The app stores downloaded GGUF models and multimodal projectors under `~/.cache/app_models/` by default, or in an Unsloth-compatible folder you pick in **Tab 1 → Model Storage**. The first model download can be very large (multi-GB); the app shows a preflight check of available disk space and asks for confirmation before starting downloads.

## Updating

No need to re-clone or rebuild your installation. From the project folder run:

```bash
./update.sh            # checks GitHub for new commits, previews them, then fast-forwards
./update.sh --check    # only report whether an update is available (changes nothing)
```

The updater:
- fetches `origin` and lists the incoming commits and changed files **before** touching anything, and asks for confirmation;
- refuses to proceed (with clear instructions) if you have uncommitted edits or diverged local commits — your work is never overwritten;
- refreshes Python dependencies inside `.venv` **only if `requirements.txt` actually changed** between versions — no unnecessary reinstalls;
- leaves your virtual environment, downloaded models (default cache or Unsloth folder) and app configuration untouched.

Afterwards just start the app again with `./run.sh`.

## Usage Overview

| Tab | Purpose |
| --- | --- |
| **Hardware & Models** | Select GPUs, watch VRAM/RAM meters, pick model + quantization, "Download & Verify" |
| **Inference Parameters** | Raw `llama-server` flags, Temperature/Top-P/Top-K/Min-P sliders, Reasoning Budget, KV-cache quantization |
| **Captioning Presets** | MOSS-Image / Booru-SDXL / Custom prompts, Explicit Mode toggle, Export/Import config JSON |
| **Dataset Explorer** | Pick dataset directory (recursive option), browse images, zoom/pan viewer |
| **Batch Processing** | Select folders/images, Start Auto-Caption / Stop / Save All, progress bar + live `llama-server` log |

The **global side panel** always shows the focused image's caption editor (loads existing `.txt`/`.json` captions) with a manual "Save Caption" button.

## Troubleshooting

- **"qt.qpa.plugin: Could not load the Qt platform plugin xcb"** → install the optional system libraries above, or force Wayland with `QT_QPA_PLATFORM=wayland ./run.sh`.
- **No GPUs detected** → the app still starts; GPU-dependent features are disabled with a clear message in the Hardware tab. Check that `nvidia-smi` or `rocm-smi` works in a terminal.
- **`llama-server` not found** → build llama.cpp (`cmake --build build --config Release -t llama-server`) and add its `bin` directory to your `PATH`, or point the app to the binary in the Inference Parameters tab.
- **OOM during batch runs** → the app automatically kills the server, reduces context size (`-c`) by 1024, restarts, and logs a warning.

## License

See [LICENSE](LICENSE).
