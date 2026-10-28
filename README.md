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
| **Inference Parameters** | Raw `llama-server` flags, Temperature/Top-P/Top-K/Min-P sliders, **Qwen3.8 reasoning controls** (Enable-Thinking master switch; effort combo exposing exactly what the Unsloth-patched chat template supports distinctly — `xhigh` / `medium` / `low`, never the silently-aliased `high`; Reasoning Budget slider with an Unlimited (`-1`) checkbox; Preserve-Thinking toggle), KV-cache quantization, multi-GPU tensor-split mode (`layer` default — the llama.cpp-recommended split — / `row` / `graph`, passed as `-sm`), **API Key** field (sent as `--api-key` to a spawned server and as `Authorization: Bearer` on every request) and auto host/port reachability check with attach-to-remote semantics (see below). The reasoning knobs travel **per request** (`chat_template_kwargs.enable_thinking` / `reasoning_effort` / `preserve_thinking` + top-level `reasoning_budget`), so changing them mid-conversation applies to the next turn without restarting `llama-server`; `--jinja` is added automatically at launch (required for the kwargs to reach the template) and `--reasoning-budget` only when a numeric cap > 0 is set |
| **Captioning Presets** | MOSS-Image / Booru-SDXL / Custom prompts, Explicit Mode toggle, Export/Import config JSON |
| **Dataset Explorer** | Pick dataset directory (recursive option), browse images, zoom/pan viewer |
| **Batch Processing** | Select folders/images, Start Auto-Caption / Stop / Save All, **Batch Save Extant Captions** (re-saves every existing sidecar of the checked images in the active preset's format — MOSS JSON → `.json`, tags → cleaned `.txt` — no server round-trip; images without a caption are skipped), progress bar + live `llama-server` log; clicking anywhere on an image row toggles its checkbox; **"Verbose conversation log"** checkbox dumps the full llama-server exchange into the Log Output: every prompt sent (system + user text, base64 images shown as size placeholders), streamed answer deltas, thinking/reasoning output (`delta.reasoning_content`) tagged `[thinking]`, raw SSE chunks, finish reason and token usage — all prefixed with `[llama]` |

The **global side panel** always shows the focused image's caption editor (loads existing `.txt`/`.json` captions) with a manual "Save Caption" button.

### Startup configuration override (`captioner_config.json`)

At launch the app looks for an optional `captioner_config.json` and, if found,
imports it automatically — overriding the hardwired preset defaults as well as
the persisted Tab 1/Tab 2 settings, storage location and model selection before
any tab is built. Search order (first existing file wins):

1. `$APPCAP_CONFIG` — explicit path to the JSON file, or a directory containing `captioner_config.json`
2. `captioner_config.json` in the current working directory
3. `captioner_config.json` in the project root (next to `run.sh`)

Both the unified bundle produced by *Captioning Presets → Export full
application config* and legacy presets-only files are accepted; unknown keys
are skipped with warnings and a corrupt file never blocks startup (the import
result and any warnings are written to the Log Output). To pin a machine's
configuration, export once next to the launcher and every start afterwards
replays it.

### Connecting to a server: API key & host reachability (Tab 2)

**API Key** — Tab 2 has an *API Key* field (password-masked, with a Show
toggle; persisted in the app settings). It is used in both directions:

* when the app **starts** `llama-server` locally it is forwarded as
  `--api-key`, so the spawned server demands that token;
* when the app **talks** to a server (single captioning, batch runs, health
  probes) it is sent as the `Authorization: Bearer <key>` header. Leaving the
  field empty falls back to the `MOSS_API_KEY` / `APPCAP_API_KEY` environment
  variables.

**Host check on commit** — after you finish editing the *Host* or *Port*
field (or press the **Check** button), a background probe answers
`GET /health` against `http://host:port` without ever blocking the UI:

* **Local host** (`localhost`, `127.x.x.x`, `::1`, `0.0.0.0`, …): purely
  advisory — the status label tells you whether something already answers
  there, but the default flow stays unchanged: you keep starting the server
  yourself from Tab 2 with the configured options, bound to the address you
  typed (`0.0.0.0` = all interfaces; clients then connect via `127.0.0.1`).
  If a local server is found running, *Start Server* reuses it and *Stop*
  only detaches instead of killing it.
* **Remote host**: if reachable and answering, *Start Server* switches into
  attach mode — it joins the existing remote server (no local process is
  launched, and a 401 even on the unauthenticated probe counts as "running,
  auth required"). Host reachable but silent → orange hint; unreachable →
  red status with the network error. A wrong/missing API key shows up as
  *auth-required* (orange) or *auth-failed* (red) respectively.

The same probe distinguishes `running` / `running-auth` / `auth-required` /
`auth-failed` / `down` / `unreachable` and logs each result to the Log
Output; repeated commits of an unchanged endpoint are deduplicated, and
*Stop* on an attached (foreign) server always detaches rather than killing
a process the app did not spawn.

### Fuzzy JSON schema handling

Models like ToriiGate-0.5 frequently deviate from the strict MOSS schema in
their responses — e.g. `watermark` as a nested per-panel dict
(`{"panel_6": {"text": "01.26"}, "page_title": "..."}`) instead of a list of
`{box_2d, label}`, or an out-of-spec `bounding_boxes` section with 0–1
coordinates and `null` panels. The parser harvests every region-like object
recursively so no data is silently dropped:

* `watermark` → normalised to the classic `[{box_2d?, label}]` list; textual
  watermarks without boxes keep their text as `label`.
* `bounding_boxes` / `extras` → kept as extra sections in the saved sidecar,
  each entry `{id, box_2d}` where `id` records the JSON path (e.g.
  `bounding_boxes.panel_1.face`). Float 0–1 coords are auto-scaled to the
  0–1000 convention; integer coords pass through unchanged.
* `null` panels, malformed boxes (≠4 numbers), and stray strings never crash
  the batch; balanced-brace extraction also fixes truncation when reasoning
  prose surrounds the JSON object.

## Troubleshooting

- **"qt.qpa.plugin: Could not load the Qt platform plugin xcb"** → install the optional system libraries above, or force Wayland with `QT_QPA_PLATFORM=wayland ./run.sh`.
- **No GPUs detected** → the app still starts; GPU-dependent features are disabled with a clear message in the Hardware tab. Check that `nvidia-smi` or `rocm-smi` works in a terminal.
- **`llama-server` not found** → build llama.cpp (`cmake --build build --config Release -t llama-server`) and add its `bin` directory to your `PATH`, or point the app to the binary in the Inference Parameters tab.
- **OOM during batch runs** → the app automatically kills the server, reduces context size (`-c`) by 1024, restarts, and logs a warning.

## License

See [LICENSE](LICENSE).
