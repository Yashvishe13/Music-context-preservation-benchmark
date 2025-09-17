
from typing import Dict, Any
import numpy as np

def tempo_and_beats_librosa(audio_path: str, sr: int = 22050):
    import librosa
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time', trim=True)
    # Handle both scalar and array tempo values properly
    if hasattr(tempo, '__len__') and len(tempo) > 0:
        tempo_val = float(tempo[0])  # Take first element if array
    else:
        tempo_val = float(tempo)  # Direct conversion if scalar
    return tempo_val, beats

def beat_fmeasure_mireval(ref_beats, est_beats) -> Dict[str, float]:
    try:
        import mir_eval
    except Exception as e:
        return {"error": f"mir_eval not available: {e}"}
    scores = mir_eval.beat.evaluate(ref_beats, est_beats)
    return {k: float(v) for k, v in scores.items()}

def downbeats_madmom(audio_path: str, beats_per_bar=(4,), fps: int = 100):
    """
    Stable madmom downbeat extraction:
    - Disable DBN threshold clipping to avoid boundary fluctuations
    - Use RNN's downbeat probability to estimate phase and cycle relabel to make downbeat=1
    - Remove incomplete bars at beginning/end (drop both ends)
    Returns: times (all beat seconds), is_downbeat (boolean vector aligned with times)
    """
    try:
        import collections, numpy as np
        if not hasattr(collections, 'MutableSequence'):
            import collections.abc
            collections.MutableSequence = collections.abc.MutableSequence
            collections.Iterable = collections.abc.Iterable
            collections.Mapping = collections.abc.Mapping
            collections.MutableMapping = collections.abc.MutableMapping
            collections.Sequence = collections.abc.Sequence
        if not hasattr(np, 'float'):
            np.float = float; np.int = int; np.complex = complex; np.bool = bool

        from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor

        # ---- Fix process (avoid numpy 2.x aggregation errors & unified no threshold clipping) ----
        def _safe_process(self, activations, **kwargs):
            import itertools as it, numpy as _np
            if not _np.asarray(activations).any():
                return _np.empty((0, 2))
            results = []
            for hmm, obs in zip(self.hmms, it.repeat(activations)):
                path, log_prob = hmm.viterbi(obs)
                log_prob = float(_np.asarray(log_prob).ravel()[0])
                results.append((path, log_prob))
            best = max(range(len(results)), key=lambda i: results[i][1])
            path, _ = results[best]
            st = self.hmms[best].transition_model.state_space
            om = self.hmms[best].observation_model
            positions = st.state_positions[path]
            beat_numbers = positions.astype(int) + 1
            # Peak alignment (not dependent on threshold)
            beats = _np.empty(0, dtype=int)
            beat_range = om.pointers[path] >= 1
            idx = _np.nonzero(_np.diff(beat_range.astype(int)))[0] + 1
            if beat_range[0]:
                idx = _np.r_[0, idx]
            if beat_range[-1]:
                idx = _np.r_[idx, beat_range.size]
            if idx.any():
                for left, right in idx.reshape((-1, 2)):
                    peak = _np.argmax(activations[left:right]) // 2 + left
                    beats = _np.hstack((beats, peak))
            return _np.vstack(((beats / float(self.fps)),
                               beat_numbers[beats])).T

        DBNDownBeatTrackingProcessor.process = _safe_process

        # ---- RNN activation ----
        rnn = RNNDownBeatProcessor()
        act = rnn(audio_path)  # (num_frames, 2) [beat_prob, downbeat_prob]
        import numpy as np
        act = np.asarray(act, dtype=np.float32)
        act = np.nan_to_num(act, nan=0.0, posinf=1.0, neginf=0.0)
        act = np.clip(act, 0.0, 1.0, out=act)
        if float(act.max()) <= 0.0:
            return np.array([]), np.array([], dtype=bool)

        # ---- DBN decoding (threshold=0 disable clipping; fixed candidate time signatures) ----
        tracker = DBNDownBeatTrackingProcessor(
            beats_per_bar=list(beats_per_bar),
            fps=fps,
            num_threads=1,
            threshold=0.0,
            correct=True
        )
        beats = tracker(act)  # (N,2): [time_sec, beat_no]
        if beats.size == 0:
            return np.array([]), np.array([], dtype=bool)

        times = beats[:, 0]
        beat_nums = beats[:, 1].astype(int)
        B = int(np.max(beat_nums)) if beat_nums.size else int(beats_per_bar[-1])

        # ---- Phase normalization: use downbeat probability to determine which beat is 1 ----
        frame_idx = np.clip(np.rint(times * fps).astype(int), 0, len(act) - 1)
        down_probs = act[frame_idx, 1]
        means = [down_probs[beat_nums == k].mean() if np.any(beat_nums == k) else -np.inf
                 for k in range(1, B + 1)]
        k_star = int(np.argmax(means) + 1)
        beat_nums = ((beat_nums - k_star) % B) + 1  # Make k_star → 1

        # ---- Remove incomplete bars at both ends: need B-1 beats before and after ----
        is_downbeat = (beat_nums == 1)
        db_idx = np.flatnonzero(is_downbeat)
        good = []
        for idx in db_idx:
            has_left  = (idx - (B - 1)) >= 0
            has_right = (idx + (B - 1)) < len(beat_nums)
            if has_left and has_right:
                good.append(idx)
        mask = np.zeros_like(is_downbeat, dtype=bool)
        if good:
            mask[np.array(good, dtype=int)] = True

        # Debug (can be commented out)
        print(f"madmom: beats={len(times)}, downbeats(full-bars)={int(mask.sum())}, B={B}, phase={k_star}")
        print("times:", times)
        print("beat_nums:", beat_nums)

        return times, mask

    except Exception as e:
        try:
            info = f" act: type={type(act)}, dtype={getattr(act,'dtype',None)}, shape={getattr(act,'shape',None)}"
        except Exception:
            info = ""
        print(f"madmom failed: {e}.{info}")
        return None, None


def _estimate_beat_period_from_all_beats(all_beats_times: np.ndarray) -> float:
    """Estimate single beat duration using median adjacent difference of all beat points; fallback to 0.5s (120BPM)."""
    if all_beats_times is None or len(all_beats_times) < 2:
        return 0.5
    d = np.diff(all_beats_times)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else 0.5

def _best_offset_grid(ref_times: np.ndarray, est_times: np.ndarray,
                      beat_period: float, tol: float) -> tuple[float, int]:
    """Grid search within a small range of ±0.5*beat_length to find the offset that maximizes hit count."""
    if ref_times.size == 0 or est_times.size == 0:
        return 0.0, 0
    # First estimate using median of nearest neighbor differences
    diffs = []
    for t in ref_times:
        j = int(np.argmin(np.abs(est_times - t)))
        diffs.append(est_times[j] - t)
    diffs = np.asarray(diffs, float)
    good = np.abs(diffs) <= 0.6 * beat_period
    delta0 = float(np.median(diffs[good])) if np.any(good) else 0.0

    grid = np.linspace(delta0 - 0.5 * beat_period, delta0 + 0.5 * beat_period, 41)

    def match_cnt(delta):
        i = j = 0; tp = 0
        e_shift = est_times + delta
        while i < len(ref_times) and j < len(e_shift):
            d = e_shift[j] - ref_times[i]
            if abs(d) <= tol:
                tp += 1; i += 1; j += 1
            elif d < -tol:
                j += 1
            else:
                i += 1
        return tp

    counts = np.array([match_cnt(d) for d in grid])
    best_idx = int(np.argmax(counts))
    return float(grid[best_idx]), int(counts[best_idx])

def downbeat_alignment_mireval(ref_path: str, est_path: str,
                               base_tol: float = 0.07) -> dict:
    """
    Use mir_eval for downbeat F1 evaluation (align offset first, then evaluate; adaptive tolerance).
    Depends on: your downbeats_madmom(audio_path) -> (all_beats_times, is_downbeat_bool)
    """
    try:
        import mir_eval
    except Exception as e:
        return {"downbeat_f1": 0.0, "error": f"mir_eval not available: {e}"}

    # 1) Extract downbeat time series (standardized phase & removed incomplete bars)
    rt_all, r_isdb = downbeats_madmom(ref_path)
    et_all, e_isdb = downbeats_madmom(est_path)
    if rt_all is None or et_all is None:
        return {"downbeat_f1": 0.0, "error": "madmom failed"}

    r_db = rt_all[r_isdb]
    e_db = et_all[e_isdb]

    # 2) Estimate beat length & tolerance (≥70ms and ≥ 0.1 × beat_length)
    beat_period = _estimate_beat_period_from_all_beats(rt_all)
    tol = max(float(base_tol), 0.10 * beat_period)

    # 3) Evaluate only within overlapping time window, with half-beat margin (more stable)
    if r_db.size == 0 or e_db.size == 0:
        return {"downbeat_f1": 0.0, "tol_seconds": float(tol), "n_ref": int(r_db.size), "n_est": int(e_db.size)}

    start = max(r_db[0], e_db[0]) + 0.5 * beat_period
    end   = min(r_db[-1], e_db[-1]) - 0.5 * beat_period
    r_db = r_db[(r_db >= start) & (r_db <= end)]
    e_db = e_db[(e_db >= start) & (e_db <= end)]
    if r_db.size == 0 or e_db.size == 0:
        return {"downbeat_f1": 0.0, "tol_seconds": float(tol), "n_ref": int(r_db.size), "n_est": int(e_db.size), "reason": "no_overlap"}

    # 4) Estimate & fine-tune global offset, then use mir_eval's f_measure for scoring
    delta, cnt = _best_offset_grid(r_db, e_db, beat_period, tol)

    f1 = mir_eval.beat.f_measure(r_db, e_db + delta, f_measure_threshold=tol)

    # 5) If still near 0, try second tier with wider tolerance (≥90ms and ≥0.2×beat_length)
    widened = False
    if f1 < 1e-6:
        tol2 = max(tol, 0.20 * beat_period, 0.09)
        delta2, _ = _best_offset_grid(r_db, e_db, beat_period, tol2)
        f1_2 = mir_eval.beat.f_measure(r_db, e_db + delta2, f_measure_threshold=tol2)
        if f1_2 > f1:
            f1 = f1_2
            tol = tol2
            delta = delta2
            widened = True

    return {
        "downbeat_f1": float(f1),
        "offset_seconds": float(delta),
        "tol_seconds": float(tol),
        "used_widen_tol": bool(widened),
        "n_ref": int(r_db.size),
        "n_est": int(e_db.size),
    }

import numpy as np

def downbeat_alignment_accuracy(ref_path: str, est_path: str, base_tol: float = 0.07):
    """
    Alignment based only on downbeats:
      - Estimate global time offset Δ (nearest neighbor median, note sign! use -median(e - r))
      - Grid fine-tune within Δ±0.75*beat_length
      - Two-pointer one-to-one matching (allow insertion/deletion)
      - Use mode of beat sequence index differences for constant offset to filter anomalies
    Returns recall based on ref.
    """
    rt, rmask = downbeats_madmom(ref_path)
    et, emask = downbeats_madmom(est_path)
    if rt is None or et is None:
        return {"downbeat_acc": 0.0, "error": "madmom failed"}

    if len(rt) < 2:
        return {"downbeat_acc": 0.0, "reason": "not_enough_beats"}
    beat_period = float(np.median(np.diff(rt)))
    tol = max(float(base_tol), 0.10 * beat_period)  # ≥70ms and ≥0.1 beat

    # Downbeat times and their indices in all beat points
    r_idx_all = np.flatnonzero(rmask)
    e_idx_all = np.flatnonzero(emask)
    r_db_all = rt[r_idx_all]
    e_db_all = et[e_idx_all]
    if r_db_all.size == 0 or e_db_all.size == 0:
        return {"downbeat_acc": 0.0, "tol_seconds": float(tol), "n_ref": int(r_db_all.size), "n_est": int(e_db_all.size)}

    # Evaluate only in overlapping window (leave half-beat margin)
    start = max(r_db_all[0], e_db_all[0]) + 0.5 * beat_period
    end   = min(r_db_all[-1], e_db_all[-1]) - 0.5 * beat_period
    r_keep = (r_db_all >= start) & (r_db_all <= end)
    e_keep = (e_db_all >= start) & (e_db_all <= end)
    r_db = r_db_all[r_keep]; r_idx = r_idx_all[r_keep]
    e_db = e_db_all[e_keep]; e_idx = e_idx_all[e_keep]
    if r_db.size == 0 or e_db.size == 0:
        return {"downbeat_acc": 0.0, "tol_seconds": float(tol), "n_ref": int(r_db.size), "n_est": int(e_db.size), "reason": "no_overlap"}

    # —— Key correction: sign of Δ —— #
    # diffs = e - r, want e + Δ ≈ r, then Δ ≈ -median(diffs)
    diffs = []
    for t in r_db:
        j = int(np.argmin(np.abs(e_db - t)))
        diffs.append(e_db[j] - t)
    diffs = np.asarray(diffs, float)
    good = np.abs(diffs) <= 0.75 * beat_period
    delta0 = -float(np.median(diffs[good])) if np.any(good) else 0.0  # ← Take negative sign!

    # Grid fine-tune centered on Δ0 (±0.75*beat_length)
    grid = np.linspace(delta0 - 0.75 * beat_period, delta0 + 0.75 * beat_period, 61)

    def match_pairs(delta, tol_):
        i = j = 0
        pairs = []
        e_shift = e_db + delta
        while i < len(r_db) and j < len(e_shift):
            d = e_shift[j] - r_db[i]
            if abs(d) <= tol_:
                pairs.append((i, j)); i += 1; j += 1
            elif d < -tol_:
                j += 1
            else:
                i += 1
        return pairs

    # Find best Δ under initial tolerance
    best_delta = 0.0
    best_pairs = []
    for d in grid:
        pairs = match_pairs(d, tol)
        if len(pairs) > len(best_pairs):
            best_pairs = pairs; best_delta = float(d)

    # If still 0, widen tolerance once (to 20% beat or ≥90ms)
    widened = False
    if len(best_pairs) == 0:
        tol2 = max(tol, 0.20 * beat_period, 0.09)
        for d in grid:
            pairs = match_pairs(d, tol2)
            if len(pairs) > len(best_pairs):
                best_pairs = pairs; best_delta = float(d); widened = True
        tol = tol2

    if len(best_pairs) == 0:
        return {
            "downbeat_acc": 0.0,
            "tol_seconds": float(tol),
            "offset_seconds": float(best_delta),
            "used_widen_tol": bool(widened),
            "n_ref": int(len(r_db)),
            "n_est": int(len(e_db)),
            "matched_pairs": 0
        }

    # Use mode of beat sequence index differences to filter local insertions/deletions
    offsets = np.array([e_idx[j] - r_idx[i] for (i, j) in best_pairs], dtype=int)
    vals, counts = np.unique(offsets, return_counts=True)
    mode_off = int(vals[np.argmax(counts)])
    tp = int(np.sum(offsets == mode_off))
    recall = float(tp / max(1, len(r_db)))

    return {
        "downbeat_acc": recall,
        "tol_seconds": float(tol),
        "offset_seconds": float(best_delta),
        "index_offset_mode": int(mode_off),
        "used_widen_tol": bool(widened),
        "n_ref": int(len(r_db)),
        "n_est": int(len(e_db)),
        "matched_pairs": int(len(best_pairs))
    }

def rhythm_score(audio_ref: str, audio_est: str) -> Dict[str, Any]:
    out = {}
    t0, b0 = tempo_and_beats_librosa(audio_ref)
    t1, b1 = tempo_and_beats_librosa(audio_est)
    out["tempo_ref_bpm"] = t0
    out["tempo_est_bpm"] = t1
    out["delta_bpm"] = float(abs(t0 - t1))
    out["beat_mir_eval"] = beat_fmeasure_mireval(b0, b1)
    # out["downbeat_alignment"] = downbeat_alignment_accuracy(audio_ref, audio_est)
    # out["downbeat_alignment"] = downbeat_alignment_mireval(audio_ref, audio_est)
    return out

if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser(description="Harmony and Tonality Score between two audio files")
    p.add_argument("--ref", required=True, help="Reference audio file (treated as 'truth')")
    p.add_argument("--est", required=True, help="Edited audio file (to evaluate)")
    args = p.parse_args()
    result = rhythm_score(args.ref, args.est)
    import json
    print(json.dumps(result, indent=2))
