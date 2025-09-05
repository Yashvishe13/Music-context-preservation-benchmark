
from typing import Dict, Any, Tuple
import numpy as np

def f0_librosa(audio_path: str, sr: int = 22050, fmin: float = 55.0, fmax: float = 1760.0, hop_length: int = 256):
    import librosa, numpy as np
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    f0, vflag, vprob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    return times, f0, vflag

def f0_essentia(audio_path: str, sr: int = 44100):
    try:
        import essentia.standard as ess, numpy as np
    except Exception as e:
        return None, None, None
    loader = ess.MonoLoader(filename=audio_path, sampleRate=sr)
    y = loader()
    mel = ess.PredominantPitchMelodia()
    f0, conf = mel(y)
    times = np.arange(len(f0)) / 44100.0 * 128
    vflag = f0 > 0
    f0_hz = np.where(vflag, f0, np.nan)
    return times, f0_hz, vflag

def mir_melody_scores(t_ref, f0_ref, v_ref, t_est, f0_est, v_est) -> Dict[str, float]:
    try:
        import mir_eval, numpy as np
    except Exception as e:
        return {"error": f"mir_eval not available: {e}"}
    fr = np.where(np.isnan(f0_ref), 0.0, f0_ref)
    fe = np.where(np.isnan(f0_est), 0.0, f0_est)
    scores = {
        "overall_accuracy": mir_eval.melody.overall_accuracy(v_ref.astype(bool), fr, v_est.astype(bool), fe),
        "raw_pitch_accuracy": mir_eval.melody.raw_pitch_accuracy(v_ref.astype(bool), fr, v_est.astype(bool), fe),
        "raw_chroma_accuracy": mir_eval.melody.raw_chroma_accuracy(v_ref.astype(bool), fr, v_est.astype(bool), fe),
        "voicing_recall": mir_eval.melody.voicing_recall(v_ref.astype(bool), v_est.astype(bool)),
        "voicing_false_alarm": mir_eval.melody.voicing_false_alarm(v_ref.astype(bool), v_est.astype(bool)),
    }
    return {k: float(v) for k, v in scores.items()}

def contour_dtw_distance(t_ref, f0_ref, t_est, f0_est) -> Dict[str, float]:
    from scipy.interpolate import interp1d
    import numpy as np
    if len(t_ref)==0 or len(t_est)==0:
        return {"pitch_dtw_cosine": 0.0}
    t0 = max(t_ref[0], t_est[0]); t1 = min(t_ref[-1], t_est[-1])
    if t1 <= t0 + 1e-6:
        return {"pitch_dtw_cosine": 0.0}
    grid = np.linspace(t0, t1, int((t1-t0)*50)+1)
    def interp(t, f):
        fz = np.nan_to_num(f, nan=0.0)
        return interp1d(t, fz, kind="nearest", fill_value="extrapolate")(grid)
    r = interp(t_ref, f0_ref); e = interp(t_est, f0_est)
    eps = 1e-8
    def to_cents(x):
        return 1200*np.log2(np.maximum(x, eps)/440.0)
    R = to_cents(r).reshape(-1,1); E = to_cents(e).reshape(-1,1)
    dR = np.diff(R, axis=0); dE = np.diff(E, axis=0)
    
    # Check if we have valid data for DTW
    if len(dR) == 0 or len(dE) == 0:
        return {"pitch_dtw_cosine": 0.0}
    
    try:
        from cp_metrics.utils import dtw_cosine
        sim = dtw_cosine(dR, dE)
        # Handle NaN results
        if np.isnan(sim) or np.isinf(sim):
            return {"pitch_dtw_cosine": 0.0}
        return {"pitch_dtw_cosine": float(sim)}
    except Exception as e:
        return {"pitch_dtw_cosine": 0.0}

def interval_tokens(t, f0) -> list:
    import numpy as np
    if len(t) < 2: return []
    grid = np.linspace(t[0], t[-1], int((t[-1]-t[0])*20)+1)
    from scipy.interpolate import interp1d
    f = interp1d(t, np.nan_to_num(f0, nan=0.0), kind="nearest", fill_value="extrapolate")(grid)
    cents = 1200*np.log2(np.maximum(f, 1e-8)/440.0)
    steps = np.round(np.diff(cents)/100.0).astype(int)
    return [int(s) for s in steps]

def motif_ngram_overlap(t_ref, f0_ref, t_est, f0_est, n: int = 3) -> Dict[str, float]:
    A = interval_tokens(t_ref, f0_ref)
    B = interval_tokens(t_est, f0_est)
    def grams(xs):
        return {tuple(xs[i:i+n]) for i in range(len(xs)-n+1)} if len(xs) >= n else set()
    GA, GB = grams(A), grams(B)
    if not GA:
        return {"jaccard": 0.0, "recall": 0.0}
    inter = len(GA & GB); uni = len(GA | GB)
    return {"jaccard": inter/(uni+1e-8), "recall": inter/(len(GA)+1e-8)}

def melody_score(audio_ref: str, audio_est: str, use_essentia: bool = False) -> Dict[str, Any]:
    if use_essentia:
        tr, fr, vr = f0_essentia(audio_ref)
        te, fe, ve = f0_essentia(audio_est)
    else:
        tr, fr, vr = f0_librosa(audio_ref)
        te, fe, ve = f0_librosa(audio_est)
    if tr is None or te is None:
        return {"error": "F0 extraction failed (try toggling use_essentia)"}
    out = {}
    out["mir_melody"] = mir_melody_scores(tr, fr, vr, te, fe, ve)
    out["contour_dtw"] = contour_dtw_distance(tr, fr, te, fe)
    out["motif_overlap_n3"] = motif_ngram_overlap(tr, fr, te, fe, n=3)
    return out
