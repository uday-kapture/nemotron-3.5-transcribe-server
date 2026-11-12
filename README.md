# nemo-3.5-transcribe

Generate training data from conversational audio with
[NVIDIA Nemotron 3.5 Transcribe](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B):

- **Per-speaker transcripts** with word timestamps, from the model's diarization-conditioned speaker tags.
- **Turn-taking labels** for every utterance: interruption (barge-in), failed interruption, backchannel,
  anticipated turn end, simultaneous start, clean switch, continuation. These are for training smaller
  transcribers and voicebots that handle interruptions well.
- **Rich tags** from the model: non-speech events (`<ev:laughter>`, `<ev:breath>`, ...) and speaking style
  spans (`<sty:stt:fear_urgency>` ... `<sty:end:fear_urgency>`), attached to the speaker who produced them.

The model itself has **no interruption or overlap tokens**. Its vocabulary is `<spk:N>`, 10 `<ev:*>` and
11 `<sty:*>` tags. Turn-taking labels are therefore derived here from per-speaker word timing. See
[Labels](#turn-taking-labels).

## How it works

```
audio ──► vLLM server (GPU) ──► raw/<id>.<pass>.json ──► annotate (CPU) ──► conversations/ + manifests/
          two prompts per file:
          speaker: <spk:N> tags + CTC word timestamps
          rich:    <ev:*>/<sty:*> tags + timestamps
```

1. **transcribe** sends each file twice, once per [model-card prompt](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B#prompt-guide),
   with `capture_ctc_timestamps` on. Raw responses are cached, so re-running resumes where it stopped.
2. **annotate** groups each speaker's words into inter-pausal units (IPUs), labels how each IPU starts
   relative to the other speakers, and moves the rich-pass tags onto whoever was speaking at that time.
   It also scores word agreement between the two passes (`pass_agreement`), which works as a cheap
   filter for low-quality transcripts.

## Setup

**GPU host (inference).** Follow the model card's vLLM setup (NeMo Speech `main` + vLLM 0.32.0, one H100 80GB
or similar), download the checkpoint, then:

```bash
MODEL_DIR=./NVIDIA-Nemotron-3.5-Transcribe-30B-A4B scripts/serve.sh
```

The script is the card's command with the CTC timestamp middleware enabled and MTP2 speculative decoding
(highest batch throughput on H100 per the card). The DFlash2 draft model is faster but disables word
timestamps, so it can't be used here.

**Client (anywhere, CPU only).**

```bash
uv sync
```

## Usage

```bash
# Both stages
uv run nemo-3-5-transcribe run /data/calls --out out/ --server http://gpu-host:8000

# Or separately: transcribe once, re-annotate as often as you like while tuning thresholds
uv run nemo-3-5-transcribe transcribe /data/calls --out out/ --server http://gpu-host:8000 --concurrency 32
uv run nemo-3-5-transcribe annotate --out out/ --config interactions.example.toml
```

Supported inputs: `.wav .flac .mp3 .ogg .opus .m4a`, up to 1 hour and 180 MB each (server limits).
`--passes speaker` skips the rich pass.

## Outputs

| Path | One row per | Use |
| --- | --- | --- |
| `manifests/segments.jsonl` | speaker segment (≤ `--max-segment-s`, default 20 s) | ASR training (NeMo manifest: `audio_filepath`, `offset`, `duration`, `text`) |
| `manifests/speakers.jsonl` | recording × speaker | per-person transcripts with timed segments |
| `manifests/conversations.jsonl` | recording | full `<spk:N>`-serialised transcript, tagged and plain |
| `conversations/<id>.json` | recording | everything: IPUs, word timings, labels, events, styles, raw diarization |

Segment rows also carry `text_tagged` (with `<int:*>`, `<ev:*>`, `<sty:*>` inline), `onset`, and `overlap`.
`overlap` is true when another speaker is audible in the clip; filter on it for clean single-speaker
ASR data. Example tagged conversation:

```
<spk:0> thank you for calling how can I help you today <spk:1> <int:interruption> sorry but my order has not arrived <ev:laughter>
```

## Turn-taking labels

Each IPU gets one `onset` label. The thresholds are in `InteractionConfig` and can be overridden with
`--config` (see `interactions.example.toml`).

| Label | Rule |
| --- | --- |
| `switch` | starts after the previous speaker stopped (`gap_s` = response latency) |
| `continue` | same speaker resumes after a pause |
| `interruption` | starts inside another speaker's IPU and that speaker stops first. `yield_s` = how long they kept talking |
| `failed_interruption` | starts inside another speaker's IPU, but that speaker outlasts it |
| `backchannel` | short (≤ 1.2 s, ≤ 3 words) and the other speaker keeps the floor. In a pause, it also needs every word in the lexicon (`yeah`, `mm-hmm`, `haan`, `ji`, `achha`, ...) |
| `overlap_turn_end` | starts in the last 0.6 s of another speaker's IPU (anticipated end, not disruptive) |
| `simultaneous_start` | both started within 0.2 s |

The interrupted IPU records `cut_off_by`. For voicebots, `interruption` vs `backchannel` is the key
distinction: whether to stop talking or keep going.

## Caveats

- Labels are only as good as diarization and CTC word timing in overlapped speech. Spot-check a
  hand-labelled sample of your own calls and tune the thresholds before training on the output.
- Diarization is strongest in the model's 13 diarization languages. Other languages still transcribe,
  but speaker tags are less reliable.
- The speaker prompt names `<spk:0>`–`<spk:3>`. Recordings with more than four speakers are out of scope
  for the default prompt.
- `truncated: true` means generation hit `--max-tokens`. Re-run those files with a higher limit.

## Development

```bash
uv run pytest
uvx ruff check src tests && uvx ruff format src tests
```
