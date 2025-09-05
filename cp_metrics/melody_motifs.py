# melodic_official.py
from typing import Dict, Tuple, Optional
import numpy as np

# -----------------------
# 1) F0 提取（官方实现）
# -----------------------
def f0_librosa(audio_path: str,
               sr: int = 22050,
               fmin: float = 55.0,
               fmax: float = 1760.0,
               hop_length: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import librosa
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    f0, vflag, _vprob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
    t = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    v = (vflag.astype(bool)) if vflag is not None else ~np.isnan(f0)
    return t.astype(float), f0.astype(float), v.astype(bool)

def f0_essentia(audio_path: str, sr: int = 44100) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        import essentia.standard as ess
    except Exception:
        return None, None, None
    loader = ess.MonoLoader(filename=audio_path, sampleRate=sr)
    y = loader()
    mel = ess.PredominantPitchMelodia()  # 默认 hopSize=128
    f0, conf = mel(y)
    # 更稳：从算法对象读取 hopSize（若不可用则退回 128）
    hop = mel.paramValue('hopSize') if hasattr(mel, 'paramValue') else 128
    t = np.arange(len(f0)) * (hop / float(sr))
    v = (f0 > 0)
    f0_hz = np.where(v, f0, np.nan)
    return t.astype(float), f0_hz.astype(float), v.astype(bool)

# ---------------------------------------
# 2) 对齐到公共时间网格（实现细节，非指标）
#    - 仅做线性/最近邻重采样，保证 mir_eval 输入等长
# ---------------------------------------
def _resample_to_grid(t: np.ndarray, y: np.ndarray, grid: np.ndarray, kind: str = "linear") -> np.ndarray:
    # y 可能有 NaN（非发声），线性插值前先把 NaN 暂时当 0 处理
    y_safe = np.nan_to_num(y, nan=0.0)
    if kind == "linear":
        return np.interp(grid, t, y_safe)
    elif kind == "nearest":
        # 最近邻（不依赖 scipy）
        idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
        left, right = t[idx - 1], t[idx]
        choose_left = (grid - left) <= (right - grid)
        out_idx = idx.copy()
        out_idx[choose_left] = idx[choose_left] - 1
        return y_safe[out_idx]
    else:
        raise ValueError("unsupported kind")

def _align_on_common_grid(t_ref: np.ndarray, f0_ref: np.ndarray, v_ref: np.ndarray,
                          t_est: np.ndarray, f0_est: np.ndarray, v_est: np.ndarray,
                          frame_rate: float = 100.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # 取重叠窗
    t0 = max(float(t_ref[0]), float(t_est[0]))
    t1 = min(float(t_ref[-1]), float(t_est[-1]))
    if not (t1 > t0):
        return np.array([]), np.array([]), np.array([]), np.array([])

    step = 1.0 / frame_rate
    # 避免末端浮点问题，+1 让右端点能覆盖
    grid = np.arange(t0, t1 + 1e-9, step, dtype=float)

    # f0 用线性插值；voicing 用最近邻（0/1）
    fr = _resample_to_grid(t_ref, f0_ref, grid, kind="linear")
    fe = _resample_to_grid(t_est, f0_est, grid, kind="linear")
    vr = _resample_to_grid(t_ref, v_ref.astype(float), grid, kind="nearest") >= 0.5
    ve = _resample_to_grid(t_est, v_est.astype(float), grid, kind="nearest") >= 0.5

    # mir_eval 规范：未发声帧 F0 置 0.0
    fr = np.where(vr, fr, 0.0)
    fe = np.where(ve, fe, 0.0)
    return fr.astype(float), vr.astype(bool), fe.astype(float), ve.astype(bool)

# ---------------------------------------
# 3) 官方指标（仅 mir_eval.melody）
# ---------------------------------------
def melody_metrics_official(audio_ref: str,
                            audio_est: str,
                            extractor: str = "librosa") -> Dict[str, float]:
    import mir_eval

    if extractor == "essentia":
        tr, fr, vr = f0_essentia(audio_ref)
        te, fe, ve = f0_essentia(audio_est)
        if tr is None or te is None:
            # 回退 librosa
            tr, fr, vr = f0_librosa(audio_ref)
            te, fe, ve = f0_librosa(audio_est)
    else:
        tr, fr, vr = f0_librosa(audio_ref)
        te, fe, ve = f0_librosa(audio_est)

    if tr is None or te is None or len(tr) == 0 or len(te) == 0:
        return {"error": "F0 extraction failed"}

    # 对齐到公共网格（保证等长 & 对齐）
    fr_g, vr_g, fe_g, ve_g = _align_on_common_grid(tr, fr, vr, te, fe, ve, frame_rate=100.0)
    if fr_g.size == 0:
        return {"error": "No temporal overlap after alignment"}

    # —— 下面 5 个即 mir_eval 官方旋律指标 —— #
    out = {
        "overall_accuracy": float(mir_eval.melody.overall_accuracy(vr_g, fr_g, ve_g, fe_g)),
        "raw_pitch_accuracy": float(mir_eval.melody.raw_pitch_accuracy(vr_g, fr_g, ve_g, fe_g)),
        "raw_chroma_accuracy": float(mir_eval.melody.raw_chroma_accuracy(vr_g, fr_g, ve_g, fe_g)),
        "voicing_recall": float(mir_eval.melody.voicing_recall(vr_g, ve_g)),
        "voicing_false_alarm": float(mir_eval.melody.voicing_false_alarm(vr_g, ve_g)),
    }
    return out

if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--est", required=True)
    p.add_argument("--extractor", choices=["librosa", "essentia"], default="librosa")
    args = p.parse_args()
    scores = melody_metrics_official(args.ref, args.est, extractor=args.extractor)
    print(json.dumps(scores, indent=2, ensure_ascii=False))
