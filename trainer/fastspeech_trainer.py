
from utils.plot import plot_spectrogram, plot_alignment
import logging
import os
from trainer.base_runners import BaseTrainer
import tensorflow as tf
import tensorflow.keras.mixed_precision.experimental as mixed_precision
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class FastSpeechTrainer(BaseTrainer):

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
        super(FastSpeechTrainer, self).__init__(config=config)
        # define metrics to aggregates data and use tf.summary logs them
        self.T=config['num_mels']//2
        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.steps = steps
        self.epochs = epochs
        self.is_mixed_precision = is_mixed_precision
        self.set_strategy(strategy)

    def set_train_metrics(self):
        self.train_metrics = {
            "mel_loss_before": tf.keras.metrics.Mean("train_ctc_loss", dtype=tf.float32),
            "mel_loss_after": tf.keras.metrics.Mean("train_att_loss", dtype=tf.float32),
            "duration_loss": tf.keras.metrics.Mean("train_alig_loss", dtype=tf.float32),
        }

    def set_eval_metrics(self):
        self.eval_metrics = {
            "mel_loss_before": tf.keras.metrics.Mean("eval_ctc_loss", dtype=tf.float32),
            "mel_loss_after": tf.keras.metrics.Mean("eval_att_loss", dtype=tf.float32),
            "duration_loss": tf.keras.metrics.Mean("eval_alig_loss", dtype=tf.float32),
        }


    def init_train_eval_metrics(self, list_metrics_name):
        """Init train and eval metrics to save it to tensorboard."""
        self.train_metrics = {}

        for name in list_metrics_name:
            self.train_metrics.update(
                {name: tf.keras.metrics.Mean(name='train_' + name, dtype=tf.float32)}
            )

    def reset_states_train(self):
        """Reset train metrics after save it to tensorboard."""
        for metric in self.train_metrics.keys():
            if metric != 'ratio':
                self.train_metrics[metric].reset_states()

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
        charactor, duration, mel, tacotron_mel, speaker_id= batch

        with tf.GradientTape() as tape:
            masked_mel_before, masked_mel_after, masked_duration_outputs = self.model(
                charactor,
                attention_mask=tf.math.not_equal(charactor, 0),
                speaker_ids=speaker_id,
                duration_gts=duration,
                training=True,
            )

            duration_loss = self.mask_mse_duration(duration, masked_duration_outputs)
            mel_loss_before = self.mask_mse_mel(tacotron_mel, masked_mel_before)+self.mask_mse_mel(tacotron_mel[:,:,self.T:], masked_mel_before[:,:,self.T:])
            mel_loss_after = self.mask_mse_mel(mel, masked_mel_after)+self.mask_mse_mel(mel[:,:,self.T:], masked_mel_after[:,:,self.T:])
            train_loss = duration_loss + mel_loss_before + mel_loss_after
            if self.is_mixed_precision:
                scaled_train_loss = self.optimizer.get_scaled_loss(train_loss)
        if self.is_mixed_precision:
            scaled_gradients = tape.gradient(scaled_train_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(train_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_metrics["duration_loss"].update_state(duration_loss)
        self.train_metrics["mel_loss_before"].update_state(mel_loss_before)
        self.train_metrics["mel_loss_after"].update_state(mel_loss_after)
        return masked_mel_after

    @tf.function(experimental_relax_shapes=True)
    def _eval_step(self, batch):
        """Train model one step."""
        charactor, duration, mel, tacotron_mel,speaker_id = batch
        masked_mel_before, masked_mel_after, masked_duration_outputs = self.model(
            charactor,
            attention_mask=tf.math.not_equal(charactor, 0),
            speaker_ids=speaker_id,
            duration_gts=duration,
            training=False,
        )
       
        duration_loss = self.mask_mse_duration(duration, masked_duration_outputs)
        mel_loss_before = self.mask_mse_mel(tacotron_mel, masked_mel_before)
        mel_loss_after = self.mask_mse_mel(mel, masked_mel_after)
        self.eval_metrics["duration_loss"].update_state(duration_loss)
        self.eval_metrics["mel_loss_before"].update_state(mel_loss_before)
        self.eval_metrics["mel_loss_after"].update_state(mel_loss_after)
        return masked_mel_after

    def mask_mse_duration(self, y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, 0), 1)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        loss = tf.square(y_pred - y_true)

        loss = mask * loss
        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)

        return loss

    def mask_mse_mel(self, y_true, y_pred):
        T1=tf.shape(y_true)[1]
        T2=tf.shape(y_pred)[1]
        T=tf.reduce_min([T1,T2])
        y_true=y_true[:,:T,:]
        y_pred=y_pred[:,:T,:]
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mask = tf.cast(tf.not_equal(y_true, -(self.config['max_abs_value']+0.1)), 1)
        loss = tf.square(y_pred - y_true)
        loss = mask * loss
        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
        return loss


    def plot_result(self,pred,target):

        os.makedirs(os.path.join(self.config['outdir'], 'plots'), exist_ok=True)

        plot_spectrogram(pred,
                         os.path.join(self.config['outdir'], 'plots','mel-before-{}.png'.format(self.steps)),
                         target_spectrogram=target)
    def _eval_batches(self):
        """One epoch evaluation."""

        for metric in self.eval_metrics.keys():
            self.eval_metrics[metric].reset_states()
        n=0
        for batch in self.eval_datasets:

            charactor, duration, mel,_, speaker_id = batch
            pred=self.strategy.run(self._eval_step,args=(batch,))
            n+=1
            self.eval_progbar.update(1)
            # Print eval info to progress bar
            self._print_eval_metrics(self.eval_progbar)
            if n>=self.eval_steps_per_epoch:
                break
        self.plot_result(pred[0].numpy(),mel[0].numpy())
        self._write_to_tensorboard(self.eval_metrics, self.steps, stage="eval")

    def fit(self, epoch=None):
        if epoch is not None:
            self.epochs = epoch
            self.train_progbar.set_description_str(
                f"[Train] [Epoch {epoch}/{self.config['num_epochs']}]")
        self._train_batches()

        self._check_eval_interval()