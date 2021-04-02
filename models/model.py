from models.tacotron2 import Tacotron2Config,TFTacotron2
from utils.text_featurizers import TextFeaturizer
from utils.speech_featurizers import SpeechFeaturizer
from models.vocoder import *
from models.fastspeech import *
import logging
import os
class TTSmodel():
    def __init__(self, config=None,vocoder_config=None):
        assert config is not None or vocoder_config is not None,'must one config'
        if config is not None:
            self.config = config
            self.acoustic=config['model_name']
        else:
            self.config = None
        self.vocoder_config=vocoder_config

        if vocoder_config is not None:
            self.GL = SpeechFeaturizer(vocoder_config).inv_mel_spectrogram
            self.vocoder_type=vocoder_config['vocoder_model']
        else:
            self.GL = SpeechFeaturizer(config).inv_mel_spectrogram
            self.vocoder_type=None

        if self.config is not None:
            self.text_featurizer=TextFeaturizer(config)

    def load_model(self,training=True):
        if self.config is not None:
            if self.acoustic=='Tacotron2':
                self.config['vocab_size']=self.text_featurizer.num_classes
                tac_config=Tacotron2Config(**self.config)
                self.acoustic_model=TFTacotron2(tac_config,training)
            elif self.acoustic=='FastSpeech':
                self.config['vocab_size']=self.text_featurizer.num_classes
                fast_config=FastSpeechConfig(**self.config)
                self.acoustic_model=TFFastSpeech(fast_config)
        if self.vocoder_config is not None:
            if self.vocoder_type =='MelGan':
                melgan_config=MelGANGeneratorConfig(**self.vocoder_config)
                self.vocoder=TFMelGANGenerator(melgan_config)
            elif self.vocoder_type=='MultiGen':
                multi_config=MultiGeneratorConfig(**self.vocoder_config)
                self.vocoder=TFMultiWindowGenerator(multi_config)
            else:
                raise ValueError('vocoder type not support.')
        if training and self.vocoder_type is not None:
            if self.vocoder_config['use_gan']:
                self.discriminator=TFMelGANMultiScaleDiscriminator(MelGANDiscriminatorConfig(**self.vocoder_config))

        if not training:
            assert self.config is not None
            self.acoustic_model._build()
            if self.vocoder_config is not None:
                self.vocoder._build()
            self.load_checkpoint()

    def load_checkpoint(self,):
        """Load checkpoint."""
        self.checkpoint_dir = os.path.join(self.config["outdir"], "checkpoints")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.acoustic_model.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
        logging.info('acoustic load model at {}'.format(os.path.join(self.checkpoint_dir, files[-1])))
        if self.vocoder_config is not None:
            self.checkpoint_dir = os.path.join(self.vocoder_config["outdir"], "checkpoints")
            files = os.listdir(self.checkpoint_dir)
            files= [i for i in files if 'g' in i]
            files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
            self.vocoder.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
            logging.info('vocoder load model at {}'.format(os.path.join(self.checkpoint_dir, files[-1])))
    def synthesize(self,text,spk):
        if self.config['model_name']=='Tacotron2':
            inp=self.text_featurizer.extract(text)
            input_length=len(inp)
            spk_id=self.text_featurizer.spker_map[spk]
            inp=np.array(inp,'int32').reshape([1,-1])
            input_length=np.array(input_length,'int32').reshape([1])
            spk_id=np.array([spk_id,0],'int32').reshape([1,-1])
            decoder_output, mel_outputs, stop_token_prediction, alignment_history=self.acoustic_model.inference(input_ids=inp,
                                                                                                                input_lengths=input_length,
                                                                                                                speaker_ids=spk_id,
                                                                                                                use_window_mask=False,
                                                                                                                win_front=5,
                                                                                                                win_back=5,
                                                                                                                maximum_iterations=100,

                                                                                                                )
        else:

            inp = self.text_featurizer.extract(text)

            spk_id = self.text_featurizer.spker_map[spk]
            inp = np.array(inp, 'int32').reshape([1, -1])

            spk_id = np.array(spk_id, 'int32').reshape([1, 1])
            decoder_output, mel_outputs,duration_pred=self.acoustic_model.inference(inp,tf.math.not_equal(inp, 0),spk_id)
        if self.vocoder_config is not None:

            wav=self.vocoder(mel_outputs)
            wav=wav[0].numpy().flatten()
        else:
            wav=self.GL(mel_outputs[0].numpy().T)
        return wav