from models.fastspeech import TFFastSpeech
from models.melgan import TFMelGANGenerator,TFMultiWindowGenerator,TFMelGANMultiScaleDiscriminator
from configs import FastCombineHparams
from utils.stft import TFMultiResolutionSTFT
from utils.dataloader import Tacotron_Vocoder_DataLoader
from utils.plot import plot_spectrogram, plot_alignment
import soundfile as sf
from utils import audio
import os
import tensorflow as tf
import pypinyin
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']='2'
class FastSpeechSynthesizer():
    def __init__(self,vocoder='GL'):
        self.vocoder_type=vocoder

        self.hp=FastCombineHparams
        self.hp.embedding_dropout_prob=0.
        self.hp.encoder_conv_dropout_rate=0.
        self.hp.postnet_dropout_rate=0.
        self.compile()
        # self.load_checkpoint(os.path.join(self.hp.save_path,'model/model.h5'))
        if self.vocoder_type!='GL':
            self.load_checkpoint('./fastspeech-log-256/fastspeech/model.h5','./fastspeech-log-256/vocoder/model.h5')

        else:
            self.load_checkpoint('./fastspeech-log-256/fastspeech/model.h5')
        self._pad=0.


    def compile(self,):

        config=self.hp.FastSpeechConfig()
        # config.vocab_size=108
        print(self.hp.symbols)
        self.model = TFFastSpeech(config)
        self.symbol_to_id, self.id_to_symbol=self.get_tokens()

        if self.vocoder_type=='MelGan':
            config=FastCombineHparams
            self.vocoder=TFMelGANGenerator(config.MelGANGeneratorConfig())
        elif self.vocoder_type=='Multi':
            config = FastCombineHparams
            self.vocoder_window=config.MultiGeneratorConfig().window
            self.vocoder = TFMultiWindowGenerator(config.MultiGeneratorConfig())
        else:
            self.vocoder=audio.inv_mel_spectrogram


    def load_checkpoint(self,path,vp=None):
        self.model._build()
        self.model.load_weights(path)
        if self.vocoder_type!='GL':
            self.vocoder._build()
            self.vocoder.load_weights(vp)

    def get_tokens(self, _characters=['‘', 'K', 'e', 'zj', 'P', 'zh', 'zk', '2', 'zd', 'R', 'zb', 'zy', 'o', '…', '9', '·', 'ze', 'n', '!', 'zl', '–', 's', "'", 'zi', 'V', 'H', '?', '.', 'z4', 'zw', '3', 'z1', 'z2', '8', '"', 'zp', 'O', 'E', '4', 'zq', '0', 'v', 'zx', 'zc', 'C', 'g', '1', 'm', 'N', 'A', 'd', ' ', 'zr', 'zo', 'Z', 't', '7', 'Q', 'zkl', 'zu', 'w', 'J', 'D', 'f', 'F', 'S', 'u', 'zf', 'X', 'zn', 'r', 'G', 'Y', 'b', 'zg', 'i', 'zm', 'a', 'za', 'L', 'y', 'x', 'I', 'q', ';', '、', 'U', 'zv', 'M', '5', 'k', 'c', 'z', 'z3', '-', 'zz', 'j', 'p', 'l', 'zs', 'T', 'h', ',', 'zt', '6', 'B', ':', 'W']):
        _pad = '|'
        _eos = '~'
        symbols = [_pad, _eos] + list(_characters)
        symbol_to_id = {s: i for i, s in enumerate(symbols)}
        id_to_symbol = {i: s for i, s in enumerate(symbols)}
        return symbol_to_id, id_to_symbol

    def _pad_input(self, x, length):
        return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self._pad)

    def get_sentences(self,text):
        def is_chinese(ch):
            if '\u4e00' <= ch <= '\u9fff':
                return True
            else:
                return False
        sentences = []

        for i in text:
            sen=[]
            i=i.strip()
            for n in i:
                if is_chinese(n):
                    pys = pypinyin.lazy_pinyin(n, 8)
                    if isinstance(pys,list):
                        pys=pys[0]
                    pys=list(pys)
                    for p in pys:
                        sen.append('z'+p)
                else:
                    sen.append(n)
            # sen.append('~')
            sen='_'.join(sen)
            sentences.append(sen)
        return sentences
    def text_to_sequence(self, text):
        # text = text.split(' ')
        text = text.split('_')
        # print(text,len(text))
        sequence = self.symbols_to_sequence(list(text))
        # print(self.symbol_to_id['zw'])
        sequence.append(self.symbol_to_id['~'])
        return sequence

    def _should_keep_symbol(self, s):
        return s in self.symbol_to_id and s is not '_' and s is not '~'
    def _prepare_inputs(self, inputs):
        max_len = max([len(x) for x in inputs])
        return np.stack([self._pad_input(x, max_len) for x in inputs])
    def symbols_to_sequence(self, text):
        return [self.symbol_to_id[s] for s in text if self._should_keep_symbol(s)]
    def process_sentence(self,words):
        inputs=[]
        input_lengths=[]
        texts=self.get_sentences(words)
        for i in texts:
            # pys=pypinyin.lazy_pinyin(i,8)
            # print(i)
            sequence=np.array(self.text_to_sequence(i))
            # print(sequence)
            inputs.append(sequence)
            input_lengths.append(len(sequence))
        inputs=self._prepare_inputs(inputs)
        input_lengths=np.array(input_lengths)
        input_lengths=input_lengths.reshape([-1,1])
        return inputs,input_lengths

    def _get_output_lengths(self, duration):
        # Determine each mel length by the stop token predictions. (len = first occurence of 1 in stop_tokens row wise)
        output_lengths = [np.sum(i) for i in duration]
        return output_lengths
    def synthesize(self,texts, speaker):


        inputs,input_lengths=self.process_sentence(texts)
        # print(inputs,input_lengths)
        speaker=np.array(speaker)
        s=time.time()
        masked_mel_before, masked_mel_after, masked_duration_outputs = self.model.inference(
            tf.constant(inputs,tf.int32),
            tf.constant(np.where(inputs > 0, 1., 0),tf.bool),
            tf.constant(speaker,tf.int32),
         
            

        )
        e=time.time()
        print('inputs length:',len(inputs[0]),'fastspeech cost time:',e-s)
        np.save('test.npy',masked_mel_after[0].numpy())
        # plot_spectrogram(masked_mel_after[0].numpy(), 'test.png',
        #                  )
        if self.vocoder_type=='GL':
            target_lengths = self._get_output_lengths(masked_duration_outputs)

            # Take off the batch wise padding
            mels = [mel[:target_length, :].numpy() for mel, target_length in zip(masked_mel_after, target_lengths)]
            wavs=[self.vocoder(mel.T,self.hp) for mel in mels]
        else:
            if self.vocoder_type=='Multi':
                if masked_mel_after.shape[1]%self.vocoder_window!=0:
                    masked_mel_after=masked_mel_after[:,:-int( masked_mel_after.shape[1]%self.vocoder_window)]
                _,wavs=self.vocoder(masked_mel_after)
            else:
                s=time.time()
                wavs = self.vocoder.call(masked_mel_after)[-1]
                e=time.time()
                print('mel length',len(masked_mel_after[0]),'vocoder cost',e-s)
        return wavs


if __name__ == '__main__':
    import soundfile as sf
    from pytictoc import TicToc
    import time
    synth=FastSpeechSynthesizer(vocoder='MelGan')
    _ = synth.synthesize(['这是一句十个字的句子哦哦.'], [[7]])
    wavs = synth.synthesize(['这是一句十个.'], [[1]])
    wavs = synth.synthesize(['这是一句.'], [[2]])
    with TicToc():
        wavs=synth.synthesize(['您好,快递是否能够寄递,需要根据物品来看哦.您可登录天天快递官网,点击禁忌规定,了解详情.'],[[7]])
    # a=np.load('000.npy')
    # if a.shape[1]!=256:
    #     a=a.T
    # wavs=synth.vocoder.call(np.array([a],'float32'))

    wav=np.array(wavs[0])
    wav/=np.abs(wav).max()
    sf.write('test.wav',wav,synth.hp.sample_rate)
    # print(mel[0].shape)
