from utils.plot import plot_spectrogram, plot_alignment
import logging
import os
import tensorflow.keras.mixed_precision.experimental as mixed_precision
import tensorflow as tf
from trainer.base_runners import BaseTrainer

class Tacotron2Trainer(BaseTrainer):
    """Tacotron2 Trainer class based on Seq2SeqBasedTrainer."""

    def __init__(self,
                 config,
                 steps=0,
                 epochs=0,
                 is_mixed_precision=False,
                 strategy=None,
                 ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            config (dict): Config dict loaded from yaml format configuration file.
            is_mixed_precision (bool): Use mixed precision or not.

        """
        super(Tacotron2Trainer, self).__init__(config=config)


        self.steps = steps
        self.epochs = epochs
        self.is_mixed_precision = is_mixed_precision
        self.set_strategy(strategy)
        self.teacher_forcing_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self.config['initial_learning_rate'],
            decay_steps=self.config['decay_steps'],
            end_learning_rate=self.config['end_learning_rate'],
            cycle=True,
            name="teacher_forcing_scheduler"
        )
        self.T=config['num_mels']//2
        self.mask_value=-(config['max_abs_value']+0.1)
    def set_train_metrics(self):
        self.train_metrics = {
            "mel_loss_before": tf.keras.metrics.Mean("train_mel_loss_before", dtype=tf.float32),
            "mel_loss_after": tf.keras.metrics.Mean("train_mel_loss_after", dtype=tf.float32),
            "stop_token_loss": tf.keras.metrics.Mean("train_stop_token_loss", dtype=tf.float32),
            "guided_attention_loss": tf.keras.metrics.Mean("train_guided_attention_loss", dtype=tf.float32),
            "_ratio": tf.keras.metrics.Mean("train_ratio", dtype=tf.float32),

        }

    def set_eval_metrics(self):
        self.eval_metrics = {
            "mel_loss_before": tf.keras.metrics.Mean("eval_mel_loss_before", dtype=tf.float32),
            "mel_loss_after": tf.keras.metrics.Mean("eval_mel_loss_after", dtype=tf.float32),
            "stop_token_loss": tf.keras.metrics.Mean("eval_stop_token_loss", dtype=tf.float32),
            "guided_attention_loss": tf.keras.metrics.Mean("eval_guided_attention_loss", dtype=tf.float32),
        }

    def reset_states_train(self):
        """Reset train metrics after save it to tensorboard."""
        for metric in self.train_metrics.keys():
            self.train_metrics[metric].reset_states()

    def reset_states_eval(self):
        """Reset eval metrics after save it to tensorboard."""
        for metric in self.eval_metrics.keys():
            self.eval_metrics[metric].reset_states()

    def mask_mse(self, y, pred):
        mask = tf.where(y == self.mask_value, 0., 1.)
        mask = tf.cast(mask, tf.float32)

        diff = (y - pred) ** 2
        diff *= mask
        diff = tf.reduce_mean(diff, -1)
        diff = tf.cast(diff, tf.float32)
        mse_loss = tf.keras.losses.mse(y, pred)
        hige_loss = tf.keras.losses.mse(y[:,:,self.T:], pred[:,:,self.T:])
        return diff + mse_loss+hige_loss
    def compile(self, model, opt):

        with self.strategy.scope():
            self.model = model
            self.model._build()

            try:
                self.load_checkpoint()
            except:
                logging.info('trainer resume failed')
            self.model.summary(line_length=100)

            self.optimizer = tf.keras.optimizers.get(opt)
            if self.is_mixed_precision:
                self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer, "dynamic")
        self.set_progbar()

    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, batch):
        """Train model one step."""
        charactor, char_length, mel, mel_length, stop_gts, speaker, guided_attention = batch

        with tf.GradientTape() as tape:
            mel_outputs, post_mel_outputs, stop_outputs, alignment_historys = self.model(
                charactor,
                char_length,
                speaker_ids=speaker,
                mel_outputs=mel,
                mel_lengths=mel_length,
                training=True)

            mel_loss_before = self.mask_mse(mel, mel_outputs)
            mel_loss_after = self.mask_mse(mel, post_mel_outputs)
            stop_token_loss = tf.keras.losses.binary_crossentropy(stop_gts, stop_outputs,True)
            stop_token_loss =tf.expand_dims(stop_token_loss,-1)
            # calculate guided attention loss.
            T1=tf.shape(guided_attention)[-1]
            T2=tf.shape(alignment_historys)[-1]
            T=tf.reduce_min([T1,T2])
            guided_attention=guided_attention[:,:,:T]
            attention_masks = tf.cast(tf.math.not_equal(guided_attention, 0.), tf.float32)

            loss_att = tf.reduce_sum(tf.abs(alignment_historys[:,:,:T] * guided_attention) * attention_masks,-1)
            loss_att /= tf.reduce_sum(attention_masks,-1)
            loss_att=tf.reduce_sum(loss_att,-1,keepdims=True)

            # sum all loss
            train_loss = stop_token_loss * 10. + mel_loss_before + mel_loss_after + loss_att
            train_loss = tf.nn.compute_average_loss(train_loss,
                                                    global_batch_size=self.global_batch_size)
            if self.is_mixed_precision:
                scaled_train_loss = self.optimizer.get_scaled_loss(train_loss)
        train_list=self.model.trainable_variables
        if self.config['finetune']:
            train_list=[i for i in train_list if 'encoder' not in i.name and 'speaker' not in i.name]
        if self.is_mixed_precision:
            scaled_gradients = tape.gradient(scaled_train_loss, train_list)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(train_loss, train_list)
        self.optimizer.apply_gradients(zip(gradients, train_list))

        self.train_metrics["stop_token_loss"].update_state(stop_token_loss)
        self.train_metrics["mel_loss_before"].update_state(mel_loss_before)
        self.train_metrics["mel_loss_after"].update_state(mel_loss_after)
        self.train_metrics["guided_attention_loss"].update_state(loss_att)

        self._apply_schedule_teacher_forcing()

    @tf.function(experimental_relax_shapes=True)
    def _eval_step(self, batch):
        """Train model one step."""
        charactor, char_length, mel, mel_length, stop_gts, speaker, guided_attention = batch

        mel_outputs, post_mel_outputs, stop_outputs, alignment_historys = self.model(
            charactor,
            char_length,
            speaker_ids=speaker,
            mel_outputs=mel,
            mel_lengths=mel_length,
            training=False)

        mel_loss_before = tf.keras.losses.mse(mel, mel_outputs)
        mel_loss_after = tf.keras.losses.mse(mel, post_mel_outputs)
        stop_token_loss = tf.keras.losses.binary_crossentropy(stop_gts, stop_outputs,True)
        T1 = tf.shape(guided_attention)[-1]
        T2 = tf.shape(alignment_historys)[-1]
        T = tf.reduce_min([T1, T2])
        guided_attention = guided_attention[:, :, :T]
        # calculate guided attention loss.
        attention_masks = tf.cast(tf.math.not_equal(guided_attention, 0.), tf.float32)
        loss_att = tf.reduce_sum(tf.abs(alignment_historys[:,:,:T] * guided_attention) * attention_masks)
        loss_att /= tf.reduce_sum(attention_masks)

        self.eval_metrics["stop_token_loss"].update_state(stop_token_loss)
        self.eval_metrics["mel_loss_before"].update_state(mel_loss_before)
        self.eval_metrics["mel_loss_after"].update_state(mel_loss_after)
        self.eval_metrics["guided_attention_loss"].update_state(loss_att)
        return post_mel_outputs,alignment_historys

    def _apply_schedule_teacher_forcing(self):

        if self.steps >= self.config['decay_steps']:
            # change _ratio on sampler.
            self.model.decoder.sampler._ratio = self.teacher_forcing_scheduler(
                self.steps - self.config['decay_steps'])
            self.train_metrics['_ratio'].update_state(self.model.decoder.sampler._ratio)
            if self.steps == self.config['decay_steps']:
                logging.info(f"(Steps: {self.steps}) Starting apply schedule teacher forcing.")
        else:
            self.train_metrics['_ratio'].update_state(self.model.decoder.sampler._ratio)

    def plot_result(self, mel_pred, mel_target, alig):
        os.makedirs(os.path.join(self.config['outdir'], 'plots'), exist_ok=True)

        plot_spectrogram(mel_pred,
                         os.path.join(self.config['outdir'], 'plots', 'mel-before-{}.png'.format(self.steps)),
                         target_spectrogram=mel_target)

        plot_alignment(alig, os.path.join(self.config['outdir'], 'plots', 'alig-{}.png'.format(self.steps)))

    def _train_batches(self):
        """Train model one epoch."""

        for batch in self.train_datasets:
            try:

                self.strategy.run(self._train_step,args=(batch,))
                self.steps += 1
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

            charactor, char_length, mel, mel_length, stop_gts, speaker, guided_attention = batch
            post_mel_outputs,alignment_historys = self.strategy.run(self._eval_step,args=(batch,))
            n += 1
            self.eval_progbar.update(1)
            # Print eval info to progress bar
            self._print_eval_metrics(self.eval_progbar)
            if n >= self.eval_steps_per_epoch:
                break
        self.plot_result(post_mel_outputs[0].numpy(), mel[0].numpy(),alignment_historys[0].numpy())
        self._write_to_tensorboard(self.eval_metrics, self.steps, stage="eval")

    def fit(self, epoch=None):
        if epoch is not None:
            self.epochs = epoch
            self.train_progbar.set_description_str(
                f"[Train] [Epoch {epoch}/{self.config['num_epochs']}]")
        self._train_batches()

        self._check_eval_interval()