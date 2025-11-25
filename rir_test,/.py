"""
rir_with_directivity.py

Unified interface to generate RIR:
    get_rir(room_dim, source, mic, directivity, ...)
Supports directivity = "omni", "cardioid", "bidirectional"

- omni: use pyroomacoustics.ShoeBox.compute_rir()
- cardioid / bidirectional: simple ISM + angle weighting (frequency-independent)

Dependencies:
    pip install pyroomacoustics numpy scipy matplotlib soundfile
"""
import numpy as np
from numpy.linalg import norm
import pyroomacoustics as pra
import scipy.signal as sig
import soundfile as sf
import matplotlib.pyplot as plt

c = 343.0

def _D_omni(cos_theta):
    return 1.0

def _D_cardioid(cos_theta):
    return 0.5 * (1.0 + cos_theta)

def _D_figure8(cos_theta):
    return cos_theta

_DIRECTIVITY_FUNCS = {
    "omni": _D_omni,
    "cardioid": _D_cardioid,
    "bidirectional": _D_figure8,
    "figure8": _D_figure8,
}

def _ism_rir_with_directionality(room_dim, source, mic, mic_orient,
                                  fs=16000, rir_length=1.0,
                                  reflection_coeffs=(0.8,0.8,0.7,0.7,0.6,0.6),
                                  max_order=5, directivity="cardioid"):
    """
    Manual ISM: enumerate image sources and multiply each path by D(theta).
    reflection_coeffs: (left, right, front, back, floor, ceiling)

    Note: for multiple reflections on the same axis this function uses the
    geometric-mean approximation of the two opposing wall coefficients.
    """
    if directivity not in _DIRECTIVITY_FUNCS:
        raise ValueError("directivity should be one of " + ", ".join(_DIRECTIVITY_FUNCS.keys()))
    Dfunc = _DIRECTIVITY_FUNCS[directivity]

    Lx, Ly, Lz = room_dim
    sx, sy, sz = source
    mx, my, mz = mic
    mic_orient = np.asarray(mic_orient, dtype=float)
    if norm(mic_orient) == 0:
        mic_orient = np.array([1., 0., 0.])
    mic_orient = mic_orient / norm(mic_orient)

    N = int(np.ceil(rir_length * fs))
    rir = np.zeros(N)

    for nx in range(-max_order, max_order+1):
        x_img = 2 * Lx * nx + ((-1) ** nx) * sx
        rx = abs(nx)
        rx_coeff = (reflection_coeffs[0] * reflection_coeffs[1]) ** (rx / 2.0)

        for ny in range(-max_order, max_order+1):
            y_img = 2 * Ly * ny + ((-1) ** ny) * sy
            ry = abs(ny)
            ry_coeff = (reflection_coeffs[2] * reflection_coeffs[3]) ** (ry / 2.0)

            for nz in range(-max_order, max_order+1):
                z_img = 2 * Lz * nz + ((-1) ** nz) * sz
                rz = abs(nz)
                rz_coeff = (reflection_coeffs[4] * reflection_coeffs[5]) ** (rz / 2.0)

                img_pos = np.array([x_img, y_img, z_img])
                vec = np.array([mx, my, mz]) - img_pos
                dist = norm(vec)
                if dist < 1e-8:
                    continue

                delay_s = dist / c
                idx = int(round(delay_s * fs))
                if idx >= N:
                    continue

                # geometric spreading (1 / r)
                amp = 1.0 / dist
                # multiply by per-axis reflection attenuation (approximation)
                amp *= rx_coeff * ry_coeff * rz_coeff

                # arrival direction and directivity weight
                arrival_dir = vec / dist
                cos_theta = float(np.clip(np.dot(arrival_dir, mic_orient), -1.0, 1.0))
                w = Dfunc(cos_theta)
                amp *= w

                rir[idx] += amp

    return rir

def get_rir(room_dim, source, mic, mic_orient=(1.0,0.0,0.0),
            fs=16000, rir_length=0.8, absorption=0.35, max_order=6,
            reflection_coeffs=None, directivity="omni"):
    """
    Unified API:
    - room_dim: [Lx, Ly, Lz]
    - source: (x, y, z)
    - mic: (x, y, z)
    - mic_orient: unit vector (microphone pointing direction)
    - directivity: "omni" | "cardioid" | "bidirectional"

    Behavior:
    - if directivity == "omni": uses pyroomacoustics ShoeBox (ISM) to get RIR
    - otherwise: uses the manual ISM implementation (reflection_coeffs overrides absorption)

    Returns:
      1-D numpy array: RIR (length = int(rir_length * fs))
    """
    directivity = directivity.lower()
    if directivity not in _DIRECTIVITY_FUNCS:
        raise ValueError("directivity must be one of: " + ", ".join(_DIRECTIVITY_FUNCS.keys()))

    if directivity == "omni":
        # Use pyroomacoustics to generate omni RIR (ShoeBox + ISM)
        room = pra.ShoeBox(room_dim, fs=fs,
                           materials=pra.Material(absorption),
                           max_order=max_order)
        room.add_source(source)
        mic_loc = np.array([[mic[0]], [mic[1]], [mic[2]]])
        room.add_microphone_array(pra.MicrophoneArray(mic_loc, fs=fs))
        room.compute_rir()
        rir = room.rir[0][0]  # mic 0, source 0

        # truncate or pad to requested length
        N = int(np.ceil(rir_length * fs))
        if len(rir) < N:
            rir = np.pad(rir, (0, N - len(rir)))
        else:
            rir = np.array(rir[:N])
        return rir
    else:
        # Manual ISM path. If reflection_coeffs is not provided, estimate from absorption.
        if reflection_coeffs is None:
            # simple approximation: reflection r ~= sqrt(1 - absorption)
            r = np.sqrt(1.0 - max(0.0, min(1.0, absorption)))
            reflection_coeffs = (r, r, r, r, r, r)
        return _ism_rir_with_directionality(room_dim, source, mic, mic_orient,
                                            fs=fs, rir_length=rir_length,
                                            reflection_coeffs=reflection_coeffs,
                                            max_order=max_order, directivity=directivity)

if __name__ == "__main__":
    # simple demonstration
    fs = 16000
    room_dim = [6.0, 4.0, 3.0]
    source = (2.0, 1.5, 1.2)
    mic = (3.5, 2.0, 1.2)
    mic_orient = (1.0, 0.0, 0.0)

    for d in ["omni", "cardioid", "bidirectional"]:
        rir = get_rir(room_dim, source, mic, mic_orient, fs=fs,
                      rir_length=0.8, absorption=0.35, max_order=6,
                      directivity=d)
        t = np.arange(len(rir)) / fs
        plt.plot(t, rir, label=d)

    plt.legend()
    plt.xlabel("Time (s)")
    plt.title("RIR (omni/cardioid/bidirectional)")
    plt.show()

    # save a cardioid example convolved signal
    dur = 1.0
    t_sig = np.linspace(0, dur, int(fs*dur), endpoint=False)
    src_sig = sig.chirp(t_sig, f0=200, f1=2000, t1=dur) * np.hanning(len(t_sig))
    rir_card = get_rir(room_dim, source, mic, mic_orient, fs=fs,
                       rir_length=0.8, absorption=0.35, max_order=6,
                       directivity="cardioid")
    mic_sig = sig.fftconvolve(src_sig, rir_card)[:len(src_sig) + len(rir_card) - 1]
    mic_sig /= (np.max(np.abs(mic_sig)) + 1e-12)
    sf.write("mic_cardioid.wav", mic_sig, fs)
    print("Saved mic_cardioid.wav")
