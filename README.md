
# cp_twofile_metrics

Two-audio-file test harness for **context preservation** in music editing tasks (Harmony, Rhythm, Structure, Melody, Non-target proxy).

## Install

```bash
pip install -r requirements.txt
```

> Optional deps: `essentia`, `madmom`, `msaf` (install if you need those metrics).

## Run

```bash
python -m cp_metrics.runner --ref path/to/reference.wav --est path/to/edited.wav
# Use Essentia (if installed) for key/melody:
python -m cp_metrics.runner --ref ref.wav --est est.wav --use-essentia
# Skip structure metrics if MSAF not installed:
python -m cp_metrics.runner --ref ref.wav --est est.wav --skip-structure
```

## Modules / Methods

- **Harmony / Tonality** (`cp_metrics/harmony_tonality.py`)
  - `key_scale_essentia(path)`, `key_relatedness(...)` (circle-of-fifths), 
  - `chroma_similarity(ref, est, method='cqt')`,
  - `chord_sequences_madmom(path)`, `chord_similarity_mireval(ref, est)`.
- **Rhythm / Meter** (`cp_metrics/rhythm_meter.py`)
  - `tempo_and_beats_librosa(path)`, `beat_fmeasure_mireval(ref_beats, est_beats)`,
  - `downbeats_madmom(path)`, `downbeat_alignment_accuracy(ref, est)`.
- **Structural Form** (`cp_metrics/structural_form.py`)
  - `msaf_segments(path)`, `mir_segment_metrics(...)`, `structural_score(ref, est)`.
- **Melodic Content / Motifs** (`cp_metrics/melody_motifs.py`)
  - `f0_librosa(path)` / `f0_essentia(path)`,
  - `mir_melody_scores(...)`, `contour_dtw_distance(...)`, `motif_ngram_overlap(...)`.
- **Non-target Stems** (2-file proxy) (`cp_metrics/non_target_stems.py`)
  - `proxy_instrument_mismatch(...)` placeholder; add your tagger later.
