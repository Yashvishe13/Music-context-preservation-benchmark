
import numpy as np

PITCH_TO_PC = {
    "C":0, "C#":1, "Db":1, "D":2, "D#":3, "Eb":3, "E":4, "Fb":4, "E#":5, "F":5,
    "F#":6, "Gb":6, "G":7, "G#":8, "Ab":8, "A":9, "A#":10, "Bb":10, "B":11, "Cb":11, "B#":0
}
PC_TO_NAME = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]

def _relative_major_pc(pc:int, scale:str)->int:
    if scale.lower().startswith("min"):
        return (pc + 3) % 12
    return pc % 12

def circle_of_fifths_distance(key1:str, scale1:str, key2:str, scale2:str):
    pc1 = PITCH_TO_PC.get(key1, None)
    pc2 = PITCH_TO_PC.get(key2, None)
    if pc1 is None or pc2 is None:
        return None, None
    r1 = _relative_major_pc(pc1, scale1)
    r2 = _relative_major_pc(pc2, scale2)
    steps = min([k for k in range(12) if (r1 + 7*k) % 12 == r2] + [12])
    steps = min(steps, 12 - steps)
    return steps, steps/6.0

def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = a.reshape(-1); b = b.reshape(-1)
    na = np.linalg.norm(a) + eps; nb = np.linalg.norm(b) + eps
    return float(np.dot(a,b) / (na*nb))

def dtw_cosine(X: np.ndarray, Y: np.ndarray) -> float:
    from scipy.spatial.distance import cdist
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    C = cdist(Xn, Yn, metric="cosine")
    try:
        import librosa
        _, wp = librosa.sequence.dtw(C=C)
        path = wp[::-1]
    except Exception:
        t = min(len(Xn), len(Yn)); path = [(i,i) for i in range(t)]
    sims = [1.0 - C[i,j] for (i,j) in path]
    return float(np.mean(sims)) if sims else 0.0
