
from typing import Dict, Any
import numpy as np
import sys

def key_scale_essentia(audio_path: str, sr: int = 44100) -> Dict[str, Any]:
    out = {}
    try:
        import essentia.standard as ess
        loader = ess.MonoLoader(filename=audio_path, sampleRate=sr)
        y = loader()
        key_extractor = ess.KeyExtractor()
        key, scale, strength = key_extractor(y)
        out = {"key": key, "scale": scale, "strength": float(strength)}
    except Exception as e:
        out = {"error": f"essentia not available or failed: {e}"}
    return out

def key_relatedness(ref_key: str, ref_scale: str, est_key: str, est_scale: str) -> Dict[str, Any]:
    from .utils import circle_of_fifths_distance
    steps, norm = circle_of_fifths_distance(ref_key, ref_scale, est_key, est_scale)
    if steps is None:
        return {"distance_steps": None, "distance_norm": None}
    return {"distance_steps": int(steps), "distance_norm_0to1": float(norm)}

def chroma_similarity(audio_ref: str, audio_est: str, sr: int = 22050, method: str = "cqt") -> Dict[str, float]:
    import numpy as np, librosa
    y0, sr = librosa.load(audio_ref, sr=sr, mono=True)
    y1, sr = librosa.load(audio_est, sr=sr, mono=True)
    if method == "stft":
        C0 = librosa.feature.chroma_stft(y=y0, sr=sr).T
        C1 = librosa.feature.chroma_stft(y=y1, sr=sr).T
    elif method == "cens":
        C0 = librosa.feature.chroma_cens(y=y0, sr=sr).T
        C1 = librosa.feature.chroma_cens(y=y1, sr=sr).T
    else:
        C0 = librosa.feature.chroma_cqt(y=y0, sr=sr).T
        C1 = librosa.feature.chroma_cqt(y=y1, sr=sr).T
    from .utils import cosine_sim, dtw_cosine
    dtw_sim = dtw_cosine(C0, C1)
    m0 = C0.mean(axis=0); m1 = C1.mean(axis=0)
    mean_corr = cosine_sim(m0, m1)
    return {"chroma_dtw_cosine": float(dtw_sim), "mean_chroma_cosine": float(mean_corr)}

def chord_sequences_madmom(audio_path: str):
    try:
        # Fix for madmom compatibility with Python 3.10+
        try:
            # Try to patch the collections module for madmom compatibility
            import collections
            if not hasattr(collections, 'MutableSequence'):
                import collections.abc
                collections.MutableSequence = collections.abc.MutableSequence
                collections.Iterable = collections.abc.Iterable
                collections.Mapping = collections.abc.Mapping
                collections.MutableMapping = collections.abc.MutableMapping
                collections.Sequence = collections.abc.Sequence
        except Exception as e:
            print(f"Could not apply compatibility patch: {e}")

        # Fix for NumPy compatibility
        try:
            import numpy as np
            if not hasattr(np, 'float'):
                np.float = float
                np.int = int
                np.complex = complex
                np.bool = bool
        except Exception as e:
            print(f"Could not apply NumPy compatibility patch: {e}")

        from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor
        
        feats = CNNChordFeatureProcessor()(audio_path)
        crf = CRFChordRecognitionProcessor()
        chords = crf(feats)
        intervals = np.array([[s, e] for (s, e, _) in chords], dtype=float)
        labels = [lab for (_, _, lab) in chords]
        # print("Chords found:", labels)
        # print("Intervals:", intervals)
        return intervals, labels
    except Exception as e:
        print(f"madmom chord extraction failed: {e}")
        return None, None

def chord_sequences_librosa_fallback(audio_path: str):
    """Fallback chord estimation using librosa chroma features"""
    try:
        import librosa
        import numpy as np
        
        y, sr = librosa.load(audio_path, sr=22050)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # Simple chord estimation based on chroma peak detection
        hop_length = 512
        frame_duration = hop_length / sr
        
        # Basic chord templates (major triads)
        chord_templates = {
            'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            'C#': [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            'D': [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            'D#': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            'E': [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            'F': [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            'F#': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            'G': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            'G#': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            'A': [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            'A#': [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            'B': [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        }
        
        # Estimate chord for each frame
        chord_labels = []
        for frame in chroma.T:
            best_chord = 'N'  # No chord
            best_score = 0
            for chord_name, template in chord_templates.items():
                score = np.dot(frame, template)
                if score > best_score:
                    best_score = score
                    best_chord = chord_name
            chord_labels.append(best_chord)
        
        # Create intervals (simplified - each frame as a segment)
        intervals = []
        labels = []
        current_chord = chord_labels[0] if chord_labels else 'N'
        start_time = 0.0
        
        for i, chord in enumerate(chord_labels[1:], 1):
            if chord != current_chord:
                end_time = i * frame_duration
                intervals.append([start_time, end_time])
                labels.append(current_chord)
                start_time = end_time
                current_chord = chord
        
        # Add final chord
        if intervals:
            intervals.append([start_time, len(chord_labels) * frame_duration])
            labels.append(current_chord)
        
        return np.array(intervals), labels
        
    except Exception as e:
        print(f"librosa chord estimation failed: {e}")
        return None, None

def chord_similarity_mireval(audio_ref: str, audio_est: str) -> Dict[str, float]:
    try:
        import mir_eval, numpy as np
    except Exception as e:
        return {"error": f"mir_eval not available: {e}"}
    
    # Try madmom first, fallback to librosa
    iv_r, lb_r = chord_sequences_madmom(audio_ref)
    if iv_r is None:
        print("Falling back to librosa chord estimation for reference")
        iv_r, lb_r = chord_sequences_librosa_fallback(audio_ref)
    
    iv_e, lb_e = chord_sequences_madmom(audio_est)
    if iv_e is None:
        print("Falling back to librosa chord estimation for estimate")
        iv_e, lb_e = chord_sequences_librosa_fallback(audio_est)
    
    if iv_r is None or iv_e is None or len(iv_r)==0 or len(iv_e)==0:
        return {"error": "chord extraction failed (both madmom and librosa fallback failed)"}
    
    scores = mir_eval.chord.evaluate(iv_r, lb_r, iv_e, lb_e)
    return {f"mir_{k}": float(v) for k, v in scores.items()}

def harmony_score(audio_ref: str, audio_est: str) -> Dict[str, Any]:
    res = {}
    k_ref = key_scale_essentia(audio_ref)
    k_est = key_scale_essentia(audio_est)
    res["key_ref"] = k_ref
    res["key_est"] = k_est
    if "key" in k_ref and "key" in k_est and "error" not in k_ref and "error" not in k_est:
        res["key_relatedness"] = key_relatedness(k_ref["key"], k_ref["scale"], k_est["key"], k_est["scale"])
    res["chroma_similarity"] = chroma_similarity(audio_ref, audio_est)
    # res["chord_similarity"] = chord_similarity_mireval(audio_ref, audio_est)
    return res

if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser(description="Harmony and Tonality Score between two audio files")
    p.add_argument("--ref", required=True, help="Reference audio file (treated as 'truth')")
    p.add_argument("--est", required=True, help="Edited audio file (to evaluate)")
    p.add_argument("--sr", type=int, default=22050, help="(kept for compat)")
    args = p.parse_args()
    out = harmony_score(args.ref, args.est)
    print(json.dumps(out, indent=2))
