import numpy as np
import librosa

# -------------------------
# 1) 有 stems 时：mir_eval 的 SDR/SIR/SAR（官方库）
#    museval 的 SI-SDR（可选）
# -------------------------
def bss_metrics_mir_eval(ref_stems, est_stems):
    """
    ref_stems / est_stems: shape = (n_sources, n_samples) 或 list[np.ndarray]
    返回: SDR/SIR/SAR 的均值(各源平均) + per-source（可选）
    """
    try:
        import mir_eval
    except Exception as e:
        return {"error": f"mir_eval not available: {e}"}

    ref = _as_2d_array(ref_stems)
    est = _as_2d_array(est_stems)
    nsrc = min(ref.shape[0], est.shape[0])
    L = min(ref.shape[1], est.shape[1])
    ref, est = ref[:nsrc, :L], est[:nsrc, :L]

    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(ref, est)

    out = {
        "SDR_mean": float(np.nanmean(sdr)),
        "SIR_mean": float(np.nanmean(sir)),
        "SAR_mean": float(np.nanmean(sar)),
    }
    # per-source（如不需要可去掉）
    for i in range(nsrc):
        out[f"SDR_{i}"] = float(sdr[i])
        out[f"SIR_{i}"] = float(sir[i])
        out[f"SAR_{i}"] = float(sar[i])

    # museval（SI-SDR / SI-SIR）可选
    try:
        import museval
        si_sdr = [float(museval.metrics.si_sdr(ref[i], est[i])) for i in range(nsrc)]
        out["SI_SDR_mean"] = float(np.nanmean(si_sdr))
        for i, v in enumerate(si_sdr):
            out[f"SI_SDR_{i}"] = v
    except Exception:
        out["note_museval"] = "museval not available (SI-SDR skipped)"

    return out


def _as_2d_array(x):
    if isinstance(x, np.ndarray):
        if x.ndim == 1: return x[None, :]
        if x.ndim == 2: return x
    # list of arrays
    arrs = [np.asarray(a, dtype=np.float32).ravel() for a in x]
    L = min(len(a) for a in arrs)
    return np.stack([a[:L] for a in arrs], axis=0).astype(np.float32)

# -------------------------
# 2) 没有 stems：简单谱启发式代理
# -------------------------
def proxy_extraneous_metrics(audio_ref, audio_est, sr=22050, hop=512):
    y0, sr = librosa.load(audio_ref, sr=sr, mono=True)
    y1, _  = librosa.load(audio_est, sr=sr, mono=True)
    L = min(len(y0), len(y1)); y0, y1 = y0[:L], y1[:L]

    # 对齐响度（避免能量差造成假阳性）
    y1 = _loudness_match(y0, y1)

    # Mel 频谱(归一化到 0..1)，取低/高频带能量占比
    M0 = _mel01(y0, sr, hop); M1 = _mel01(y1, sr, hop)
    T = min(M0.shape[1], M1.shape[1]); M0, M1 = M0[:, :T], M1[:, :T]

    low0 = M0[:8].mean(axis=0);  low1 = M1[:8].mean(axis=0)      # 低频带
    high0 = M0[-8:].mean(axis=0); high1 = M1[-8:].mean(axis=0)   # 高频带
    tot0 = M0.mean(axis=0) + 1e-9; tot1 = M1.mean(axis=0) + 1e-9

    # 低频突发（估计 > 参考 25% 以上的帧占比）
    lf_excess_rate = float(np.mean((low1/tot1) > 1.25*(low0/tot0)))
    # 高频过量
    hf_excess_rate = float(np.mean((high1/tot1) > 1.25*(high0/tot0)))

    # 静音错配（参考静音而估计有声的帧占比）
    rms0 = librosa.feature.rms(y=y0, hop_length=hop)[0]
    rms1 = librosa.feature.rms(y=y1, hop_length=hop)[0]
    db0 = librosa.power_to_db(rms0**2 + 1e-12, ref=np.max)
    db1 = librosa.power_to_db(rms1**2 + 1e-12, ref=np.max)
    Lf = min(len(db0), len(db1))
    vad0 = db0[:Lf] > -60.0; vad1 = db1[:Lf] > -60.0
    silence_mismatch_rate = float(np.mean((~vad0) & vad1))

    # 代理“乐器/标签不匹配率”（简单聚合，0..1，越大=越多非目标）
    mismatch_rate = float(np.clip(np.mean([lf_excess_rate, hf_excess_rate, silence_mismatch_rate]), 0, 1))

    return {
        "proxy_lf_excess_rate": lf_excess_rate,
        "proxy_hf_excess_rate": hf_excess_rate,
        "proxy_silence_mismatch_rate": silence_mismatch_rate,
        "proxy_instrument_tag_mismatch_rate": mismatch_rate
    }

def _mel01(y, sr, hop):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length=hop, power=2.0)
    Sdb = librosa.power_to_db(S, ref=np.max)
    return np.clip((Sdb + 80.0)/80.0, 0.0, 1.0)

def _loudness_match(y0, y1):
    r0 = np.sqrt(np.mean(y0**2) + 1e-12)
    r1 = np.sqrt(np.mean(y1**2) + 1e-12)
    return y1 if r1 < 1e-9 else y1 * (r0 / r1)

# -------------------------
# 3) 统一入口 + 简单总分
# -------------------------
def non_target_extraneous_score(audio_ref, audio_est, ref_stems=None, est_stems=None):
    """
    有 stems -> 报 mir_eval 的 SIR/SDR/SAR(+可选 SI-SDR) 并合成总分
    无 stems -> 只报 proxy 指标并给总分
    """
    out = {}
    have_stems = (ref_stems is not None) and (est_stems is not None)

    if have_stems:
        bss = bss_metrics_mir_eval(ref_stems, est_stems)
        out.update({f"bss_{k}": v for k, v in bss.items()})

    proxy = proxy_extraneous_metrics(audio_ref, audio_est)
    out.update(proxy)

    # 简单总分（0..1，越大=非目标干扰越少）
    # 有 stems：主看 SIR（dB）-> 线性映射到 0..1，然后和 proxy 组合
    #   映射: SIR<=0 → 0；SIR>=20 → 1（中间线性）
    def map_db_0to1(x_db): 
        return float(np.clip((x_db - 0.0)/20.0, 0.0, 1.0))

    if have_stems and "bss_SIR_mean" in out:
        sir01 = map_db_0to1(out["bss_SIR_mean"])
        proxy01 = float(1.0 - out["proxy_instrument_tag_mismatch_rate"])
        score = 0.7 * sir01 + 0.3 * proxy01
    else:
        # 无 stems：只能依赖 proxy
        score = float(1.0 - out["proxy_instrument_tag_mismatch_rate"])

    out["non_target_overall_score_0to1"] = float(np.clip(score, 0.0, 1.0))
    return out

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Compute non-target stems metrics")
    parser.add_argument("--ref", type=str, required=True, help="Reference audio file path")
    parser.add_argument("--est", type=str, required=True, help="Estimated audio file path")
    parser.add_argument("--ref_stems", type=str, help="Path to numpy file containing reference stems (shape: n_sources x n_samples)")
    parser.add_argument("--est_stems", type=str, help="Path to numpy file containing estimated stems (shape: n_sources x n_samples)")
    args = parser.parse_args()
    
    # Load stems if provided
    ref_stems = None
    est_stems = None
    if args.ref_stems and args.est_stems:
        try:
            ref_stems = np.load(args.ref_stems)
            est_stems = np.load(args.est_stems)
            print(f"Loaded stems: ref shape {ref_stems.shape}, est shape {est_stems.shape}")
        except Exception as e:
            print(f"Failed to load stems: {e}")
    
    # Compute comprehensive scores
    scores = non_target_extraneous_score(args.ref, args.est, ref_stems, est_stems)
    print(json.dumps(scores, indent=2))
