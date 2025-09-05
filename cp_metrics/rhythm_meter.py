
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

def downbeats_madmom(audio_path: str):
    try:
        # Fix for madmom compatibility with Python 3.10+
        try:
            import collections
            if not hasattr(collections, 'MutableSequence'):
                import collections.abc
                collections.MutableSequence = collections.abc.MutableSequence
                collections.Iterable = collections.abc.Iterable
                collections.Mapping = collections.abc.Mapping
                collections.MutableMapping = collections.abc.MutableMapping
                collections.Sequence = collections.abc.Sequence
        except Exception:
            pass

        # Fix for NumPy compatibility - more comprehensive patch
        import numpy as np
        if not hasattr(np, 'float'):
            np.float = float
        if not hasattr(np, 'int'):
            np.int = int  
        if not hasattr(np, 'complex'):
            np.complex = complex
        if not hasattr(np, 'bool'):
            np.bool = bool

        # Also patch the dtype attributes that madmom might use
        if not hasattr(np, 'float_'):
            np.float_ = np.float64
        if not hasattr(np, 'int_'):
            np.int_ = np.int64

        from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
        proc = RNNDownBeatProcessor()
        act = proc(audio_path)
        
        # DBNDownBeatTrackingProcessor needs fps parameter, default is 100
        tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[3,4], fps=100)
        beats = tracker(act)
        
        # Handle different return formats from madmom
        if len(beats) == 0:
            return np.array([]), np.array([], dtype=bool)
        
        # Ensure beats is a 2D array
        beats = np.atleast_2d(beats)
        if beats.shape[1] < 2:
            print(f"Warning: beats array has unexpected shape {beats.shape}")
            return np.array([]), np.array([], dtype=bool)
            
        times = beats[:, 0]
        is_down = beats[:, 1] == 1
        return times, is_down
    except Exception as e:
        print(f"madmom not available or failed: {e}")
        return None, None

def downbeat_alignment_accuracy(ref_path: str, est_path: str, tol: float = 0.07):
    ref = downbeats_madmom(ref_path)
    est = downbeats_madmom(est_path)
    if ref is None or est is None or ref[0] is None or est[0] is None: 
        return {"error": "madmom not available or failed"}
    rt, rdb = ref; et, edb = est
    if len(rt) == 0 or len(et) == 0 or rdb.sum() == 0 or edb.sum() == 0:
        return {"downbeat_acc": 0.0}
    r_times = rt[rdb]; e_times = et[edb]
    if len(r_times) == 0 or len(e_times) == 0:
        return {"downbeat_acc": 0.0}
    used = np.zeros(len(e_times), dtype=bool)
    TP = 0
    for t in r_times:
        idx = np.argmin(np.abs(e_times - t))
        if not used[idx] and abs(e_times[idx]-t) <= tol:
            used[idx] = True
            TP += 1
    return {"downbeat_acc": float(TP / max(1, len(r_times)))}

def rhythm_score(audio_ref: str, audio_est: str) -> Dict[str, Any]:
    out = {}
    t0, b0 = tempo_and_beats_librosa(audio_ref)
    t1, b1 = tempo_and_beats_librosa(audio_est)
    out["tempo_ref_bpm"] = t0
    out["tempo_est_bpm"] = t1
    out["delta_bpm"] = float(abs(t0 - t1))
    out["beat_mir_eval"] = beat_fmeasure_mireval(b0, b1)
    out["downbeat_alignment"] = downbeat_alignment_accuracy(audio_ref, audio_est)
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
