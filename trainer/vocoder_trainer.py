import logging
import os

import soundfile as sf
import tensorflow as tf
import tensorflow.keras.mixed_precision.experimental as mixed_precision

from trainer.base_runners import BaseTrainer
from utils.stft import TFMultiResolutionSTFT
import pesq

class VocoderTrainer(BaseTrainer):
    def __init__(self, config,
                 steps=0,
                 epochs=0,
                 is_mixed_precision=False,
                 strategy=None, ):
        super(VocoderTrainer, self).__init__(config=config)

        self.steps = steps
        self.epochs = epochs
        self.is_mixed_precision = is_mixed_precision
        self.set_strategy(strategy)



    def set_train_metrics(self):
        self.train_metrics = {}
        if self.config['use_gan']:
            list_metrics_name = [
                "adversarial_loss",
                "fm_loss",
                "dis_loss",
                "mels_spectrogram_loss",
                'mos',
            ]
        else:
            list_metrics_name = [
                "mels_spectrogram_loss",
                'mos'
            ]
        for name in list_metrics_name:
            self.train_metrics.update(
                {name: tf.keras.metrics.Mean(name='train_' + name, dtype=tf.float32)}
            )

    def set_eval_metrics(self):
        self.eval_metrics = {}
        if self.config['use_gan']:
            list_metrics_name = [
                "adversarial_loss",
                "fm_loss",
                "dis_loss",
                "mels_spectrogram_loss",
                'mos',
            ]
        else:
            list_metrics_name = [
                "mels_spectrogram_loss",
                'mos'
            ]
        for name in list_metrics_name:
            self.eval_metrics.update(
                {name: tf.keras.metrics.Mean(name='train_' + name, dtype=tf.float32)}
            )

    def reset_states_train(self):
        """Reset train metrics after save it to tensorboard."""
        for metric in self.train_metrics.keys():
            self.train_metrics[metric].reset_states()

    def reset_states_eval(self):
        """Reset eval metrics after save it to tensorboard."""
        for metric in self.eval_metrics.keys():
            self.eval_metrics[metric].reset_states()

    def compile(self, model_g, opt_g, model_d=None, opt_d=None):
        with self.strategy.scope():
            self.model_g = model_g
            self.model_g._build()
            self.mels_loss=TFMultiResolutionSTFT()

            self.model_g.summary(line_length=100)
            self.optimizer_g = tf.keras.optimizers.get(opt_g)
            if self.config['use_gan']:
                self.model_d = model_d
                self.model_d._build()
                self.optimizer_d = tf.keras.optimizers.get(opt_d)
            if self.is_mixed_precision:
                self.optimizer_g = mixed_precision.LossScaleOptimizer(self.optimizer_g, "dynamic")
                if self.config['use_gan']:
                    self.optimizer_d = mixed_precision.LossScaleOptimizer(self.optimizer_d, "dynamic")
            try:
                self.load_checkpoint()
            except:
                logging.info('trainer resume failed')
        self.set_progbar()

    def load_checkpoint(self, ):
        """Load checkpoint."""

        self.checkpoint_dir = os.path.join(self.config["outdir"], "checkpoints")
        files = os.listdir(self.checkpoint_dir)
        files = [i for i in files if 'g' in i]
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.model_g.load_weights(os.path.join(self.checkpoint_dir, files[-1]), by_name=True, skip_mismatch=True)

        if self.config['use_gan']:
            files = os.listdir(self.checkpoint_dir)
            files = [i for i in files if 'd' in i]
            files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
            self.model_d.load_weights(os.path.join(self.checkpoint_dir, files[-1]), by_name=True, skip_mismatch=True)

        self.steps = int(files[-1].split('_')[-1].replace('.h5', ''))

    def save_checkpoint(self, max_save=10):
        """Save checkpoint."""
        self.checkpoint_dir = os.path.join(self.config["outdir"], "checkpoints")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.model_g.save_weights(os.path.join(self.checkpoint_dir, 'model_g_{}.h5'.format(self.steps)))
        if self.config['use_gan']:
            self.model_d.save_weights(os.path.join(self.checkpoint_dir, 'model_d_{}.h5'.format(self.steps)))
        self.train_progbar.set_postfix_str("Successfully Saved Checkpoint")
        if len(os.listdir(self.checkpoint_dir)) > max_save * 2:
            files = os.listdir(self.checkpoint_dir)
            files = [i for i in files if 'g' in i]
            files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
            os.remove(os.path.join(self.checkpoint_dir, files[0]))
            if self.config['use_gan']:
                files = os.listdir(self.checkpoint_dir)
                files = [i for i in files if 'd' in i]
                files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
                os.remove(os.path.join(self.checkpoint_dir, files[0]))


    @tf.function(experimental_relax_shapes=True)
    def _eval_step(self, batch):
        """Train model one step."""
        inp, tar = batch

        if self.config['vocoder_model'] == 'MelGan':
            y_hat = self.model_g(inp, training=False)
            gen_loss = self.mels_loss([tf.squeeze(tar, -1), tf.squeeze(y_hat, -1)])
        else:
            y_hat1, y_hat2 = self.model_g(inp, training=False)  # [B, T, 1]

            gen_loss = self.mels_loss([tf.squeeze(tar, -1), tf.squeeze(y_hat1, -1)]) + self.mels_loss(
                [tf.squeeze(tar, -1), tf.squeeze(y_hat2, -1)])

        self.eval_metrics["mels_spectrogram_loss"].update_state(gen_loss)
        if self.config['vocoder_model'] == 'MelGan':
            return tar, y_hat
        else:
            return tar,  y_hat2

    def d_step(self, y, y_hat):
        p_hat = self.model_d(y_hat, training=False)
        adv_loss = 0.0
        for i in range(len(p_hat)):
            adv_loss += tf.reduce_mean(tf.keras.losses.mse(
                p_hat[i][-1], tf.ones_like(p_hat[i][-1], dtype=tf.float32)
            ),-1)
        adv_loss /= (i + 1)

        p = self.model_d(y)
        # define feature-matching loss
        fm_loss = 0.0
        for i in range(len(p_hat)):
            for j in range(len(p_hat[i]) - 1):
                fm_loss += tf.reduce_mean(tf.keras.losses.mae(
                    p_hat[i][j], p[i][j]
                ),-1)
        fm_loss /= (i + 1) * (j + 1)
        # adv_loss += fm_loss

        return adv_loss, fm_loss

    @tf.function(experimental_relax_shapes=True)
    def _one_step_generator(self, batch):
        """One step generator training."""
        mels, y = batch
        with tf.GradientTape() as g_tape:
            if self.config['vocoder_model'] == 'MelGan':
                y_hat = self.model_g(mels, training=True)  # [B, T, 1]
                adv_loss, fm_loss = self.d_step(y, y_hat)
                gen_loss = adv_loss + fm_loss + self.mels_loss([y[:, :, 0], y_hat[:, :, 0]])
            else:
                y_hat1, y_hat2 = self.model_g(mels, training=True)
                adv_loss1, fm_loss1 = self.d_step(y, y_hat1)
                adv_loss2, fm_loss2 = self.d_step(y, y_hat2)
                adv_loss = adv_loss1 * 0.5 + adv_loss2 * 0.5
                fm_loss = fm_loss1 * 0.5 + fm_loss2 * 0.5
                gen_loss = adv_loss + fm_loss + self.mels_loss([y[:, :, 0], y_hat1[:, :, 0]]) + self.mels_loss(
                    [y[:, :, 0], y_hat2[:, :, 0]])
            gen_loss = tf.nn.compute_average_loss(gen_loss,
                                                    global_batch_size=self.global_batch_size)
            if self.is_mixed_precision:
                scaled_gen_loss = self.optimizer_g.get_scaled_loss(gen_loss)

        if self.is_mixed_precision:
            scaled_gradients = g_tape.gradient(scaled_gen_loss, self.model_g.trainable_variables)
            gradients = self.optimizer_g.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = g_tape.gradient(gen_loss, self.model_g.trainable_variables)
        self.optimizer_g.apply_gradients(zip(gradients, self.model_g.trainable_variables))
        self.train_metrics["adversarial_loss"].update_state(adv_loss)
        self.train_metrics["fm_loss"].update_state(fm_loss)
        if self.config['vocoder_model'] == 'MelGan':
            self.train_metrics["mels_spectrogram_loss"].update_state(
                self.mels_loss([tf.squeeze(y, -1), tf.squeeze(y_hat, -1)]))
        else:
            self.train_metrics["mels_spectrogram_loss"].update_state(
                self.mels_loss([tf.squeeze(y, -1), tf.squeeze(y_hat1, -1)]) + self.mels_loss(
                    [tf.squeeze(y, -1), tf.squeeze(y_hat2, -1)]))

        if self.config['vocoder_model'] == 'MelGan':
            y_hat = self.model_g(mels)
            return y, y_hat
        else:
            y_hat1, y_hat2 = self.model_g(mels)
            return y, y_hat1, y_hat2

    @tf.function(experimental_relax_shapes=True)
    def _one_step_generator_stft(self, batch):
        """One step generator training."""
        mels, y = batch
        with tf.GradientTape() as g_tape:
            if self.config['vocoder_model'] == 'MelGan':
                y_hat = self.model_g(mels)
                # print(y_hat.shape)
                gen_loss = self.mels_loss([tf.squeeze(y, -1), tf.squeeze(y_hat, -1)])
            else:
                y_hat1, y_hat2 = self.model_g(mels)  # [B, T, 1]

                gen_loss = self.mels_loss([tf.squeeze(y, -1), tf.squeeze(y_hat1, -1)]) + self.mels_loss(
                    [tf.squeeze(y, -1), tf.squeeze(y_hat2, -1)])

        gradients = g_tape.gradient(gen_loss, self.model_g.trainable_variables)
        self.optimizer_g.apply_gradients(zip(gradients, self.model_g.trainable_variables))
        self.train_metrics["mels_spectrogram_loss"].update_state(gen_loss)
        if self.config['vocoder_model'] == 'MelGan':
            return y, y_hat
        else:
            return y, y_hat2

    @tf.function(experimental_relax_shapes=True)
    def _one_step_discriminator(self, y, y_hat):
        """One step discriminator training."""
        with tf.GradientTape() as d_tape:
            # y = tf.expand_dims(y, 2)
            p = self.model_d(y, training=True)
            p_hat = self.model_d(y_hat, training=True)
            real_loss = 0.0
            fake_loss = 0.0
            for i in range(len(p)):
                real_loss += tf.reduce_mean(tf.keras.losses.mse(
                    p[i][-1], tf.ones_like(p[i][-1], dtype=tf.float32)
                ),-1)
                fake_loss += tf.reduce_mean(tf.keras.losses.mse(
                    p_hat[i][-1], tf.zeros_like(p_hat[i][-1], dtype=tf.float32)
                ),-1)
            real_loss /= (i + 1)
            fake_loss /= (i + 1)
            dis_loss = real_loss + fake_loss
            dis_loss = tf.nn.compute_average_loss(dis_loss,
                                                  global_batch_size=self.global_batch_size)
            if self.is_mixed_precision:
                scaled_dis_loss = self.optimizer_d.get_scaled_loss(dis_loss)

        if self.is_mixed_precision:
            scaled_gradients = d_tape.gradient(scaled_dis_loss, self.model_d.trainable_variables)
            gradients = self.optimizer_d.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = d_tape.gradient(dis_loss, self.model_d.trainable_variables)
        self.optimizer_d.apply_gradients(zip(gradients, self.model_d.trainable_variables))
        self.train_metrics["dis_loss"].update_state(dis_loss)

    def plot_result(self, pred,target):
        os.makedirs(os.path.join(self.config['outdir'], 'plots'), exist_ok=True)

        sf.write(os.path.join(self.config['outdir'], 'plots', 'pred_{}.wav'.format(self.steps)), pred,
                 self.config['sample_rate'])
        sf.write(os.path.join(self.config['outdir'], 'plots', 'target_{}.wav'.format(self.steps)), target,
                 self.config['sample_rate'])

    def _train_batches(self):
        """Train model one epoch."""

        for batch in self.train_datasets:
            try:
                if self.config['use_gan']:
                    if self.config['vocoder_model'] == 'MelGan':
                        if self.train_metrics['mels_spectrogram_loss'].result() < self.config[
                            'mel_loss_thread'] and self.steps > self.config['gan_start_step']:
                            y, y_hat = self.strategy.run(self._one_step_generator,args=(batch,))
                        else:
                            y, y_hat = self.strategy.run(self._one_step_generator_stft,args=(batch,))
                        self.strategy.run(self._one_step_discriminator,args=(y, y_hat))
                    else:
                        if self.train_metrics['mels_spectrogram_loss'].result() < self.config[
                            'mel_loss_thread'] and self.steps > self.config['gan_start_step']:
                            y, y_hat1, y_hat = self.strategy.run(self._one_step_generator,args=(batch,))
                        else:
                            y, y_hat1, y_hat = self.strategy.run(self._one_step_generator_stft,args=(batch,))
                        # =self._one_step_generator(y,mels)
                        self.strategy.run(self._one_step_discriminator,args=(y, y_hat1,))
                        self.strategy.run(self._one_step_discriminator,args=(y, y_hat,))
                else:
                    if self.config['vocoder_model'] == 'MelGan':
                        y, y_hat = self._one_step_generator_stft(batch)
                    else:
                        y, y_hat = self._one_step_generator_stft(batch)
                self.steps += 1

                mos = pesq.pesq(self.config['sample_rate'], y[0].numpy().flatten(), y_hat[0].numpy().flatten(),'nb')
                self.train_metrics['mos'].update_state(mos)
                self.train_progbar.update(1)
                self._print_train_metrics(self.train_progbar)
                self._check_log_interval()

                if self._check_save_interval():
                    break
                if self._finished():
                    break

            except tf.errors.OutOfRangeError:
                continue
    def _eval_batches(self):
        """One epoch evaluation."""

        for metric in self.eval_metrics.keys():
            self.eval_metrics[metric].reset_states()
        n = 0
        for batch in self.eval_datasets:
            target, pred = self.strategy.run(self._eval_step, args=(batch,))
            mos=pesq.pesq(self.config['sample_rate'],target[0].numpy().flatten(),pred[0].numpy().flatten(),'nb')
            self.eval_metrics['mos'].update_state(mos)
            n += 1
            self.eval_progbar.update(1)
            # Print eval info to progress bar
            self._print_eval_metrics(self.eval_progbar)
            if n >= self.eval_steps_per_epoch:
                break
        self.plot_result(pred[0].numpy().flatten(), target[0].numpy().flatten())
        self._write_to_tensorboard(self.eval_metrics, self.steps, stage="eval")

    def fit(self, epoch=None):
        if epoch is not None:
            self.epochs = epoch
            self.train_progbar.set_description_str(
                f"[Train] [Epoch {epoch}/{self.config['num_epochs']}]")
        self._train_batches()

        self._check_eval_interval()