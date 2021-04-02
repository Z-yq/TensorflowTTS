# -*- coding: utf-8 -*-
# Copyright 2020 Minh Nguyen (@dathudeptrai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""STFT-based loss modules."""

import tensorflow as tf
import numpy as np

class TFSpectralConvergence(tf.keras.layers.Layer):
    """Spectral convergence loss."""

    def __init__(self):
        """Initialize."""
        super().__init__()

    def call(self, y_mag, x_mag):
        """Calculate forward propagation.
        Args:
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return tf.norm(y_mag - x_mag, ord="fro", axis=(-2, -1)) / tf.norm(y_mag, ord="fro", axis=(-2, -1))

class TFLogSTFTMagnitude(tf.keras.layers.Layer):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initialize."""
        super().__init__()

    def call(self, y_mag, x_mag):
        """Calculate forward propagation.
        Args:
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """

        y_mag=tf.math.log(y_mag)
        x_mag=tf.math.log(x_mag)


        return tf.losses.mae(y_mag,x_mag)

class TFSTFT(tf.keras.layers.Layer):
    """STFT loss module."""

    def __init__(self, frame_length=600, frame_step=120, fft_length=1024):
        """Initialize."""
        super().__init__()
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.spectral_convergenge_loss = TFSpectralConvergence()
        self.log_stft_magnitude_loss = TFLogSTFTMagnitude()

    def call(self, inputs):
        """Calculate forward propagation.
        Args:
            y (Tensor): Groundtruth signal (B, T).
            x (Tensor): Predicted signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value (pre-reduce).
            Tensor: Log STFT magnitude loss value (pre-reduce).
        """
        x,y=inputs
        x_mag = tf.abs(tf.signal.stft(signals=x,
                                      frame_length=self.frame_length,
                                      frame_step=self.frame_step,
                                      fft_length=self.fft_length))
        y_mag = tf.abs(tf.signal.stft(signals=y,
                                      frame_length=self.frame_length,
                                      frame_step=self.frame_step,
                                      fft_length=self.fft_length))

        x_mag = tf.math.sqrt(x_mag ** 2 + 1e-7)
        y_mag = tf.math.sqrt(y_mag ** 2 + 1e-7)

        mel_loss=self.log_stft_magnitude_loss(y_mag,x_mag)
        sc_loss = self.spectral_convergenge_loss(y_mag, x_mag)


        return mel_loss,sc_loss


class TFMultiResolutionSTFT(tf.keras.layers.Layer):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_lengths=[2048, 1024, 512,512],
                 frame_lengths=[1200, 600, 250,125],
                 frame_steps=[200, 100, 50,25],):
        """Initialize Multi resolution STFT loss module.
        Args:
            frame_lengths (list): List of FFT sizes.
            frame_steps (list): List of hop sizes.
            fft_lengths (list): List of window lengths.
        """
        super().__init__()
        assert len(frame_lengths) == len(frame_steps) == len(fft_lengths)
        self.stft_losses = []
        for frame_length, frame_step, fft_length in zip(frame_lengths, frame_steps, fft_lengths):
            self.stft_losses.append(TFSTFT(frame_length, frame_step, fft_length))

    def call(self, inputs):
        """Calculate forward propagation.
        Args:
            y (Tensor): Groundtruth signal (B, T).
            x (Tensor): Predicted signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        x,y=inputs
        sc_loss = 0.0
        mag_loss = 0.0

        for idx,f in enumerate(self.stft_losses):
            sc_l, mag_l = f([y, x])
            sc_loss += tf.reduce_mean(sc_l)
            mag_loss += tf.reduce_mean(mag_l)


        loss=sc_loss+mag_loss
        return loss


class TFMelSpectrogram(tf.keras.layers.Layer):
    """Mel Spectrogram loss."""

    def __init__(self,
                 n_mels=80,
                 f_min=80.0,
                 f_max=7600,
                 frame_length=1024,
                 frame_step=256,
                 fft_length=1024,
                 sample_rate=16000,
                 **kwargs):
        """Initialize."""
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length

        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            n_mels, fft_length // 2 + 1, sample_rate, f_min, f_max
        )

    def _calculate_log_mels_spectrogram(self, signals):
        """Calculate forward propagation.
        Args:
            signals (Tensor): signal (B, T).
        Returns:
            Tensor: Mel spectrogram (B, T', 80)
        """
        stfts = tf.signal.stft(signals,
                               frame_length=self.frame_length,
                               frame_step=self.frame_step,
                               fft_length=self.fft_length)
        linear_spectrograms = tf.abs(stfts)
        mel_spectrograms = tf.tensordot(
            linear_spectrograms, self.linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(linear_spectrograms.shape[:-1].concatenate(
            self.linear_to_mel_weight_matrix.shape[-1:]))
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)  # prevent nan.
        return log_mel_spectrograms

    def call(self, y, x):
        """Calculate forward propagation.
        Args:
            y (Tensor): Groundtruth signal (B, T).
            x (Tensor): Predicted signal (B, T).
        Returns:
            Tensor: Mean absolute Error Spectrogram Loss.
        """
        y_mag = self._calculate_log_mels_spectrogram(y)
        x_mag = self._calculate_log_mels_spectrogram(x)
        length = int(tf.shape(y_mag)[-1] // 8)
        extra=[]
        for i in range(8):
            s=length*i
            e=s+length
            extra.append(tf.reduce_mean(tf.abs(y_mag[:,:,s:e]-x_mag[:,:,s:e])))
        return tf.reduce_mean(extra)


