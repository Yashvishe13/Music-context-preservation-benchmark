# Music Context Preservation Benchmark

Two-audio-file test harness for **music context preservation** in music editing tasks (Harmony, Rhythm, Structure, Melody).

**NEW**: ⚡ Multi-core parallel batch processing for large-scale evaluations!

## Installation

```bash
pip install -r requirements.txt
```

> **Optional dependencies**: `essentia`, `madmom`, `msaf` (install if you need those specific metrics)

## Usage

### 🎵 Single File Mode

Evaluate a single reference-estimate audio pair:

```bash
# Basic usage
python -m cp_metrics.runner --ref path/to/reference.wav --est path/to/edited.wav

# With optional features
python -m cp_metrics.runner --ref ref.wav --est est.wav \
    --use-essentia \                    # Use Essentia for key/melody extraction
    --skip-structure \                   # Skip MSAF structure metrics
    --output-file results.json           # Save to file (default: stdout)
```

**Output**: JSON object with all metrics printed to stdout or saved to file.

---

### 🚀 Batch Processing Mode (Parallel)

Process multiple audio pairs in parallel using **multiprocessing** for maximum performance:

#### From JSON File

Create `pairs.json`:
```json
[
  {"id": "song1", "ref": "data/ref/song1.wav", "est": "data/edited/song1.wav"},
  {"id": "song2", "ref": "data/ref/song2.wav", "est": "data/edited/song2.wav"},
  {"id": "song3", "ref": "data/ref/song3.wav", "est": "data/edited/song3.wav"}
]
```

Run:
```bash
python -m cp_metrics.runner --batch-json pairs.json --output-dir results/
```

#### From CSV File

Create `pairs.csv`:
```csv
id,ref,est
song1,data/ref/song1.wav,data/edited/song1.wav
song2,data/ref/song2.wav,data/edited/song2.wav
song3,data/ref/song3.wav,data/edited/song3.wav
```

Run:
```bash
python -m cp_metrics.runner --batch-csv pairs.csv --output-dir results/
```

#### From Directories (Auto-matching)

Match files by filename from two directories:

```bash
python -m cp_metrics.runner \
    --ref-dir ./reference_audio/ \
    --est-dir ./edited_audio/ \
    --output-dir results/
```

Files are matched by stem name (e.g., `song1.wav` ↔ `song1.mp3`).

---

### ⚙️ Performance Options

```bash
# Auto mode (default): Uses all CPU cores automatically
python -m cp_metrics.runner --batch-json pairs.json --output-dir results/

# Manual control: Specify number of parallel workers
python -m cp_metrics.runner --batch-json pairs.json --output-dir results/ --n-workers 16

# Maximum CPU utilization (recommended for large datasets)
python -m cp_metrics.runner --batch-json pairs.json --output-dir results/ --max-cpu

# Serial mode: Process one pair at a time (for debugging)
python -m cp_metrics.runner --batch-json pairs.json --output-dir results/ --no-parallel

# Metric-level parallelism: Also parallelize metrics within each pair
# (Automatically enabled when pairs < CPU cores)
python -m cp_metrics.runner --batch-json pairs.json --output-dir results/ --parallel-metrics
```

**Performance Notes**:
- Default: Uses all available CPU cores (e.g., 128 workers on a 128-core machine)
- Each worker processes one audio pair at a time
- With 3000 pairs and 128 cores: 128 pairs process simultaneously, workers continuously pick new pairs from queue
- Metric-level parallelism uses threading for I/O-bound operations
- Expected speedup: ~Linear with CPU cores for independent pairs

---

### 🔄 Advanced Features

#### Auto-Resume (Enabled by Default)

If processing is interrupted, simply rerun the same command - already completed pairs are automatically skipped:

```bash
# First run (interrupted after 500 pairs)
python -m cp_metrics.runner --batch-json 3000_pairs.json --output-dir results/

# Resume: Automatically skips the 500 completed pairs, processes remaining 2500
python -m cp_metrics.runner --batch-json 3000_pairs.json --output-dir results/
```

To force reprocessing all pairs:
```bash
python -m cp_metrics.runner --batch-json pairs.json --output-dir results/ --no-resume
```

#### Batch Chunking (For Very Large Datasets)

Process pairs in chunks with incremental summary updates:

```bash
# Process 3000 pairs in chunks of 100 (30 batches)
python -m cp_metrics.runner \
    --batch-json 3000_pairs.json \
    --output-dir results/ \
    --batch-size 100
```

**When to use `--batch-size`:**
- ✅ **Limited RAM**: Each pair loads large audio files into memory
- ✅ **Incremental saves**: Want summary CSV updated after each chunk
- ✅ **Progress monitoring**: Easier to track progress with batch milestones
- ❌ **Not needed** for most cases: Default mode efficiently handles large datasets

**How it works:**
- Without `--batch-size`: All 3000 pairs queued, 128 workers process continuously (FASTEST)
- With `--batch-size 100`: Process 100 pairs → save summary → next 100 pairs → repeat

---

### 📊 Batch Output Structure

When running in batch mode with `--output-dir results/`, the following files are created:

```
results/
├── song1_results.json          # Full metrics for song1
├── song2_results.json          # Full metrics for song2
├── song3_results.json          # Full metrics for song3
├── summary.csv                 # Summary table with key metrics
└── errors.json                 # Error log (only if failures occur)
```

#### Summary CSV Columns

The `summary.csv` includes these key metrics:

| Column | Description |
|--------|-------------|
| `pair_id` | Identifier for the audio pair |
| `status` | `success` or `error` |
| `harmony_key_distance` | Key distance (0-1, circle of fifths) |
| `harmony_chroma_dtw` | Chroma DTW cosine similarity |
| `harmony_chroma_mean` | Mean chroma cosine similarity |
| `rhythm_tempo_delta_bpm` | Tempo difference in BPM |
| `rhythm_beat_fmeasure` | Beat tracking F-measure |
| `structure_boundary_f` | Structure boundary F-score |
| `structure_pairwise_f` | Pairwise segment F-score |
| `structure_ari` | Adjusted Rand Index |
| `melody_overall_accuracy` | Overall melody accuracy |
| `melody_raw_pitch_accuracy` | Raw pitch accuracy |
| `melody_contour_dtw` | Melodic contour DTW similarity |

#### Individual JSON Files

Each `{id}_results.json` contains:
```json
{
  "pair_id": "song1",
  "ref_path": "data/ref/song1.wav",
  "est_path": "data/edited/song1.wav",
  "status": "success",
  "error": null,
  "metrics": {
    "harmony_tonality": { ... },
    "rhythm_meter": { ... },
    "structural_form": { ... },
    "melodic_content": { ... }
  }
}
```

---

### 🎯 Example Workflows

**Small test (4 parallel workers)**:
```bash
python -m cp_metrics.runner \
    --batch-json test_pairs.json \
    --output-dir test_results/ \
    --n-workers 4 \
    --skip-structure
```

**Large-scale evaluation (3000+ pairs, maximum speed)**:
```bash
python -m cp_metrics.runner \
    --batch-json 3000_pairs.json \
    --output-dir /results/experiment1/ \
    --max-cpu \
    --use-essentia
```

**Memory-conscious processing (large audio files)**:
```bash
python -m cp_metrics.runner \
    --batch-json large_dataset.json \
    --output-dir results/ \
    --batch-size 50 \
    --n-workers 32
```

**Resume interrupted job**:
```bash
# Just rerun the same command - auto-resumes from where it stopped
python -m cp_metrics.runner \
    --batch-json 3000_pairs.json \
    --output-dir results/ \
    --max-cpu
```

**Directory matching (auto-discover pairs)**:
```bash
python -m cp_metrics.runner \
    --ref-dir /data/reference_audio/ \
    --est-dir /data/edited_audio/ \
    --output-dir /results/experiment1/
```

**Debug mode (sequential, verbose errors)**:
```bash
python -m cp_metrics.runner \
    --batch-json pairs.json \
    --output-dir debug_results/ \
    --no-parallel
```

---

### 💡 Performance Tips

**For large datasets (1000+ pairs):**
1. Use `--max-cpu` for maximum throughput
2. Use `--skip-structure` if MSAF structure metrics not needed (saves ~30% time)
3. Auto-resume is enabled by default - safe to interrupt and restart
4. Each result saved immediately - no data loss on interruption

**Estimated processing time:**
- Single pair: ~15-25 seconds (depending on metrics enabled)
- 3000 pairs on 128-core machine: ~9-12 minutes (with `--max-cpu --skip-structure`)
- Scales linearly with CPU cores

**Memory considerations:**
- Each worker loads ~2-3 audio files into RAM simultaneously
- 128 workers ≈ 256-384 audio files in memory
- If memory is limited, use `--batch-size` or reduce `--n-workers`

---

### 📋 Complete CLI Reference

#### Input Options (mutually exclusive)
| Option | Description |
|--------|-------------|
| `--ref PATH --est PATH` | Single pair mode |
| `--batch-json FILE` | Batch from JSON file |
| `--batch-csv FILE` | Batch from CSV file |
| `--ref-dir DIR --est-dir DIR` | Match files from two directories |

#### Output Options
| Option | Description |
|--------|-------------|
| `--output-dir DIR` | Output directory (batch mode, default: `./results`) |
| `--output-file FILE` | Output file (single mode, default: stdout) |

#### Processing Options
| Option | Description |
|--------|-------------|
| `--n-workers N` | Number of parallel workers (default: all CPUs) |
| `--max-cpu` | Enable maximum CPU utilization (all cores + metric parallelism) |
| `--parallel-metrics` | Parallelize metrics within each pair (threading) |
| `--no-parallel` | Disable parallelization (debug mode) |
| `--batch-size N` | Process in chunks of N pairs (for memory management) |
| `--no-resume` | Reprocess all pairs (ignore existing results) |

#### Metric Options
| Option | Description |
|--------|-------------|
| `--use-essentia` | Use Essentia for key/melody extraction |
| `--skip-structure` | Skip structure (MSAF) metrics (saves ~30% time) |

---

## Metric Modules

### Harmony / Tonality (`cp_metrics/harmony_tonality.py`)
- **Key estimation**: `key_scale_essentia(path)` - Musical key and scale detection
- **Key relatedness**: `key_relatedness(...)` - Circle-of-fifths distance
- **Chroma similarity**: `chroma_similarity(ref, est, method='cqt')` - Harmonic content comparison
- **Chord sequences**: `chord_sequences_madmom(path)` - Chord progression extraction

### Rhythm / Meter (`cp_metrics/rhythm_meter.py`)
- **Tempo & beats**: `tempo_and_beats_librosa(path)` - Tempo and beat tracking
- **Beat evaluation**: `beat_fmeasure_mireval(ref_beats, est_beats)` - Beat alignment metrics
- **Downbeats**: `downbeats_madmom(path)` - Downbeat detection
- **Alignment accuracy**: `downbeat_alignment_accuracy(ref, est)` - Downbeat F-measure

### Structural Form (`cp_metrics/structural_form.py`)
- **Segmentation**: `msaf_segments(path)` - Music structure segmentation
- **Segment metrics**: `mir_segment_metrics(...)` - Boundary and pairwise comparison
- **Overall score**: `structural_score(ref, est)` - Combined structure metrics

### Melodic Content (`cp_metrics/melody_motifs.py`)
- **F0 extraction**: `f0_librosa(path)` / `f0_essentia(path)` - Pitch contour extraction
- **Melody scores**: `mir_melody_scores(...)` - Pitch accuracy and voicing metrics
- **Contour DTW**: `contour_dtw_distance(...)` - Melodic contour similarity
- **Motif overlap**: `motif_ngram_overlap(...)` - N-gram motif similarity

---

## FAQ

**Q: I have 3000 pairs to process. Should I use `--batch-size`?**

A: Usually **no**. The default mode efficiently handles large datasets by queuing all pairs and letting workers continuously process them. Use `--batch-size` only if you have memory constraints or want incremental summary updates.

**Q: How do I resume an interrupted job?**

A: Just rerun the same command. Auto-resume is enabled by default and skips already-completed pairs (identified by existing `*_results.json` files).

**Q: How many workers should I use?**

A: Default (all CPUs) is usually best. For a 128-core machine processing 3000 pairs, all 128 cores will be utilized automatically.

**Q: What's the difference between `--n-workers` and `--parallel-metrics`?**

A: 
- `--n-workers`: Number of pairs processed in parallel (process-based, CPU-bound)
- `--parallel-metrics`: Parallelize 4 metrics within each pair (thread-based, I/O-bound)
- For best performance: Use both with `--max-cpu`

**Q: My job was killed due to memory issues. What should I do?**

A: Reduce parallel workers or use batch chunking:
```bash
python -m cp_metrics.runner --batch-json pairs.json --output-dir results/ --n-workers 32 --batch-size 50
```

**Q: Can I run multiple instances simultaneously?**

A: Yes, but use different `--output-dir` for each to avoid conflicts.

---

## Requirements

- Python 3.8+
- See `requirements.txt` for core dependencies
- Optional: `essentia`, `madmom`, `msaf` for enhanced metrics

## License

See `LICENSE` file for details.

