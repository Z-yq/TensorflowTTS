
import os
import io
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from scipy.signal import butter, lfilter
from scipy import signal
import copy

def read_raw_audio(audio, sample_rate=16000):
    if isinstance(audio, str):
        wave, _ = librosa.load(os.path.expanduser(audio), sr=sample_rate)
    elif isinstance(audio, bytes):
        wave, sr = sf.read(io.BytesIO(audio))
        wave = np.asfortranarray(wave)
        if sr != sample_rate:
            wave = librosa.resample(wave, sr, sample_rate)
    elif isinstance(audio, np.ndarray):
        return audio
    else:
        raise ValueError("input audio must be either a path or bytes")
    return wave


def normalize_audio_feature(audio_feature: np.ndarray, per_feature=False):
    """ Mean and variance normalization """
    axis = 0 if per_feature else None
    mean = np.mean(audio_feature, axis=axis)
    std_dev = np.std(audio_feature, axis=axis) + 1e-9
    normalized = (audio_feature - mean) / std_dev
    return normalized


def normalize_signal(signal: np.ndarray):
    """ Normailize signal to [-1, 1] range """
    gain = 1.0 / (np.max(np.abs(signal)) + 1e-9)
    return signal * gain


class SpeechFeaturizer:
    def __init__(self, speech_config: dict):

        # Samples
        self.speech_config=speech_config
        self.sample_rate = speech_config["sample_rate"]
        self.hop_size = int(self.sample_rate * (speech_config["hop_size"]))
        self.win_size = int(self.sample_rate * (speech_config["win_size"]))
        # Features
        self.num_mels = speech_config["num_mels"]

        self.preemphasis = speech_config["preemphasis"]

        # Normalization

    def smooth_energe(self,wav, sr):
        factor = 5
        cutoff = 20
        nyq = 0.5 * sr
        order = 3  # set low-pass filter order
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        envelop = lfilter(b, a, abs(wav))  # filter low frequency part as signal's envelop
        envelop = envelop / np.abs(envelop).max()
        envelop = envelop * factor + 1
        wav = np.divide(wav, envelop)
        wav /= np.abs(wav).max()
        return wav
    def load_wav(self,path):
        wav=read_raw_audio(path,self.sample_rate)
        wav=librosa.effects.preemphasis(wav)
        wav=self.smooth_energe(wav,self.sample_rate)
        wav=librosa.effects.trim(wav,top_db=20)[0]
        return wav

    def pad_signal(self,wavs,max_length):
        wavs = tf.keras.preprocessing.sequence.pad_sequences(wavs, max_length, 'float32', 'post', 'post')
        return wavs

    def melspectrogram(self,wav):

        D = librosa.stft(y=wav, n_fft=self.speech_config['n_fft'], hop_length=self.hop_size,
                                    win_length=self.win_size)

        assert self.speech_config['fmax'] <= self.sample_rate // 2
        mel_basis= librosa.filters.mel(self.sample_rate, self.speech_config['n_fft'], n_mels=self.num_mels
                                   , fmin=self.speech_config['fmin'], fmax=self.speech_config['fmax'])
        D= np.dot(mel_basis, np.abs(D))
        min_level = np.exp(self.speech_config['min_level_db'] / 20 * np.log(10))
        D= 20 * np.log10(np.maximum(min_level, D))
        S = D - self.speech_config['ref_level_db']

       
        S=np.clip((2 * self.speech_config['max_abs_value']) * (
                        (S - self.speech_config['min_level_db']) / (-self.speech_config['min_level_db'])) - self.speech_config['max_abs_value'],
                               -self.speech_config['max_abs_value'], self.speech_config['max_abs_value'])
            
        return S.T

    def preemphasis(self,wav):
       
        return np.append(wav[0], wav[1:] - 0.97 * wav[:-1])
       
    def inv_preemphasis(self,wav):
    
        return signal.lfilter([1], [1, -0.97], wav)
        
    def inv_mel_spectrogram(self,mel_spectrogram):
        '''Converts mel spectrogram to waveform using librosa'''
        mel_spectrogram *= self.speech_config['power']
        D=(((np.clip(mel_spectrogram, -self.speech_config['max_abs_value'],
                   self.speech_config['max_abs_value']) + self.speech_config['max_abs_value']) * -self.speech_config['min_level_db'] / (
                  2 * self.speech_config['max_abs_value']))+ self.speech_config['min_level_db'])
        D=np.power(10.0, (D) * 0.05)
        mel_basis = librosa.filters.mel(self.sample_rate, self.speech_config['n_fft'],
                                        n_mels=self.num_mels
                                        , fmin=self.speech_config['fmin'], fmax=self.speech_config['fmax'])
        _inv_mel_basis = np.linalg.pinv(mel_basis)
        S= np.maximum(1e-10, np.dot(_inv_mel_basis, D))
        spectro = copy.deepcopy(S)
        for i in range(self.speech_config['griffin_lim_iters']):
            estimated_wav = librosa.istft(spectro, hop_length=self.hop_size, win_length=self.win_size)
            est_stft = librosa.stft(y=estimated_wav, n_fft=self.speech_config['n_fft'], hop_length=self.hop_size,
                                    win_length=self.win_size)
            phase = est_stft / np.maximum(1e-8, np.abs(est_stft))
            spectro = S * phase
        estimated_wav = librosa.istft(spectro, hop_length=self.hop_size, win_length=self.win_size)
        result = np.real(estimated_wav)

        return self.inv_preemphasis(result)

    def _compute_pitch_feature(self, signal: np.ndarray) -> np.ndarray:
        pitches, _ = librosa.core.piptrack(
            y=signal, sr=self.sample_rate,
            n_fft=self.speech_config['n_fft'], hop_length=self.hop_size,
            fmin=0, fmax=int(self.sample_rate / 2), win_length=self.win_size, center=True
        )

        pitches = pitches.T

        assert self.num_mels <= self.speech_config['n_fft'] // 2 + 1, \
            "num_features for spectrogram should \
        be <= (sample_rate * window_size // 2 + 1)"

        return pitches[:, :self.num_mels]





