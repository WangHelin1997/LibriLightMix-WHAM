import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def read_scaled_wav(path, scaling_factor, start=0, end=None, sr=16000, mono=True):
    if end is not None:
        samples, sr_orig = sf.read(path, start=int(start*sr), stop=int(end*sr))
    else:
        samples, sr_orig = sf.read(path)

    if len(samples.shape) > 1 and mono:
        samples = samples[:, 0]

    if sr != sr_orig:
        samples = resample_poly(samples, sr, sr_orig)
    
    samples /= np.max(np.abs(samples)) + 1e-8
    samples *= scaling_factor
    return samples


def wavwrite_quantize(samples):
    return np.int16(np.round((2 ** 15) * samples))


def quantize(samples):
    int_samples = wavwrite_quantize(samples)
    return np.float64(int_samples) / (2 ** 15)


def wavwrite(file, samples, sr):
    """This is how the old Matlab function wavwrite() quantized to 16 bit.
    We match it here to maintain parity with the original dataset"""
    int_samples = wavwrite_quantize(samples)
    sf.write(file, int_samples, sr, subtype='PCM_16')


def fix_length(s1, s2, s3, s4, s5, tag1, tag2, tag3, tag4, tag5, fixed_len=5, sr=16000):
    # tag: start time
    # Fix length
    s1_out, s2_out, s3_out, s4_out, s5_out = np.zeros(int(sr*fixed_len)), np.zeros(int(sr*fixed_len)), np.zeros(int(sr*fixed_len)), np.zeros(int(sr*fixed_len)), np.zeros(int(sr*fixed_len))
    if s1.shape[0] < int(fixed_len*sr) - int(sr*tag1): # avoid out of shape
        s1_out[int(sr*tag1):s1.shape[0]+int(sr*tag1)] = s1
    else:
        s1_out[int(sr*tag1):] = s1[:(int(sr*fixed_len)-int(sr*tag1))]

    if s2.shape[0] < int(fixed_len*sr) - int(sr*tag2): # avoid out of shape
        s2_out[int(sr*tag2):s2.shape[0]+int(sr*tag2)] = s2
    else:
        s2_out[int(sr*tag2):] = s2[:(int(sr*fixed_len)-int(sr*tag2))]

    if s3.shape[0] < int(fixed_len*sr) - int(sr*tag3): # avoid out of shape
        s3_out[int(sr*tag3):s3.shape[0]+int(sr*tag3)] = s3
    else:
        s3_out[int(sr*tag3):] = s3[:(int(sr*fixed_len)-int(sr*tag3))]

    if s4.shape[0] < int(fixed_len*sr) - int(sr*tag4): # avoid out of shape
        s4_out[int(sr*tag4):s4.shape[0]+int(sr*tag4)] = s4
    else:
        s4_out[int(sr*tag4):] = s4[:(int(sr*fixed_len)-int(sr*tag4))]

    if s5.shape[0] < int(fixed_len*sr) - int(sr*tag5): # avoid out of shape
        s5_out[int(sr*tag5):s5.shape[0]+int(sr*tag5)] = s5
    else:
        s5_out[int(sr*tag5):] = s5[:(int(sr*fixed_len)-int(sr*tag5))]

    return s1_out, s2_out, s3_out, s4_out, s5_out


def create_wham_mixes(speaker_num, s1_samples, s2_samples, s3_samples, s4_samples, s5_samples, noise_samples):
    if speaker_num == 0:
        mix_clean = np.zeros_like(s1_samples)
        mix_noisy = noise_samples
    elif speaker_num == 1:
        mix_clean = s1_samples
        mix_noisy = noise_samples + s1_samples
    elif speaker_num == 2:
        mix_clean = s1_samples + s2_samples
        mix_noisy = noise_samples + s1_samples + s2_samples
    elif speaker_num == 3:
        mix_clean = s1_samples + s2_samples + s3_samples
        mix_noisy = noise_samples + s1_samples + s2_samples + s3_samples
    elif speaker_num == 4:
        mix_clean = s1_samples + s2_samples + s3_samples + s4_samples
        mix_noisy = noise_samples + s1_samples + s2_samples + s3_samples + s4_samples
    elif speaker_num == 5:
        mix_clean = s1_samples + s2_samples + s3_samples + s4_samples + s5_samples
        mix_noisy = noise_samples + s1_samples + s2_samples + s3_samples + s4_samples + s5_samples
    return mix_clean, mix_noisy
