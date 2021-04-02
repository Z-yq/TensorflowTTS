from models.tacotron2 import *
from models.melgan import *
from models.fastspeech import *
from configs import TacHparams,VocHparams,CombineHparams
from utils import audio
import os
import pypinyin
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']='-1'
class TacotronSynthesizer():
    def __init__(self,win_front=3,win_back=5,vocoder='GL'):
        self.vocoder_type=vocoder
        self.win_front=win_front
        self.win_back=win_back
        self.hp=TacHparams
        self.hp.embedding_dropout_prob=0.
        self.hp.encoder_conv_dropout_rate=0.
        self.hp.postnet_dropout_rate=0.
        self.hp.prenet_dropout_rate=0.
        self.compile()
        # self.load_checkpoint(os.path.join(self.hp.save_path,'model/model.h5'))
        if self.vocoder_type!='GL':
            self.load_checkpoint('./combine-log/tacotron/model.h5','./combine-log/vocoder/model.h5')

        else:
            self.load_checkpoint(os.path.join(self.hp.save_path,'model','model.h5'))
        self._pad=0.
        with open('pinyin_2_179phone.map',encoding='utf-8') as f:
            data=f.readlines()
        map_dict={}
        for line in data[2:]:
        #print(str(line))
            try:
                a,b=line.split('\t')
            except:
                content=line.split(' ')
                a=content[0]
                b=' '.join(content[1:])
            a=a.replace('[','').replace(']','')
            b=b.split(' ')[:-1]
            map_dict[a]=b
        self.map_dict=map_dict


    def compile(self,):

        config=self.hp.Tacotron2Config()
        # config.vocab_size=108
        #print(self.hp.symbols)
        self.model=TFTacotron2(config,False)
        self.symbol_to_id, self.id_to_symbol=self.get_tokens(self.hp.symbols)
        print('init vocoder')
        if self.vocoder_type=='MelGan':
            config=CombineHparams
            self.vocoder=TFMelGANGenerator(config.MelGANGeneratorConfig())
        elif self.vocoder_type=='Multi':
            config = CombineHparams
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
            pin=pypinyin.pinyin(i,8,neutral_tone_with_five=True)
            for p in pin:
                if p[0] in self.map_dict:
                    sen+=self.map_dict[p[0]]
                else:
                    if len(p[0])>1:
                        sen+=list(p[0])
                    else:
                        sen+=p
            sentences.append(sen)
        return sentences
    def text_to_sequence(self, text):
        # text = text.split(' ')
        #text = text.split(' ')
        # print(text,len(text))
        #print(text)
        sequence = self.symbols_to_sequence(text)
        #print(self.symbol_to_id['zw'])
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
        print(texts)
        for i in texts:
            # pys=pypinyin.lazy_pinyin(i,8)
            print(i)
            sequence=np.array(self.text_to_sequence(i))
            print(sequence)
            inputs.append(sequence)
            input_lengths.append(len(sequence))
        inputs=self._prepare_inputs(inputs)
        input_lengths=np.array(input_lengths)
        input_lengths=input_lengths.reshape([-1,1])
        return inputs,input_lengths

    def _get_output_lengths(self, stop_tokens):
        # Determine each mel length by the stop token predictions. (len = first occurence of 1 in stop_tokens row wise)
        output_lengths = [row.index(1) + 1 if 1 in row else len(row) for row in np.round(stop_tokens).tolist()]
        return output_lengths
    def synthesize(self,texts, speaker):
        from utils.plot import plot_alignment,plot_spectrogram

        inputs,input_lengths=self.process_sentence(texts)
        # print(inputs,input_lengths)
        speaker=np.array(speaker)

        # tacotron2 inference.
        mel_outputs, post_mel_outputs, stop_outputs, alignment_historys = self.model.inference(
            inputs,
            input_lengths,
            speaker_ids=speaker,
            use_window_mask=False,
            win_front=20,
            win_back=20,
            maximum_iterations=1000,
           
        )
        plot_spectrogram(post_mel_outputs[0].numpy(),'./mel.png','inference')
        if self.vocoder_type=='GL':
            target_lengths = self._get_output_lengths(stop_outputs)

            # Take off the batch wise padding
            mels = [mel[:target_length, :].numpy() for mel, target_length in zip(post_mel_outputs, target_lengths)]
            wavs=[self.vocoder(mel.T,self.hp) for mel in mels]
        else:
            if self.vocoder_type=='Multi':
                if post_mel_outputs.shape[1]%self.vocoder_window!=0:
                    post_mel_outputs=post_mel_outputs[:,:-int( post_mel_outputs.shape[1]%self.vocoder_window)]
                _,wavs=self.vocoder(post_mel_outputs)
            else:
                _,wavs = self.vocoder(post_mel_outputs)
        return wavs



if __name__ == '__main__':
    import soundfile as sf
    import time
    synth=TacotronSynthesizer(vocoder='GL')
    # _ = synth.synthesize(['这是一句十个字的句子'], [[0]])
    # wavs = synth.synthesize(['这是一句十个'], [[1]])
    # wavs = synth.synthesize(['这是一句'], [[2]])
    for i in range(2):
        s=time.time()
        wavs=synth.synthesize(['这是一句十个字的句子？'],[[0]])
        e=time.time()
        print(i,e-s)
    wav=np.array(wavs[0])
    wav/=np.abs(wav).max()
    sf.write('test.wav',wav,synth.hp.sample_rate)
    # print(mel[0].shape)
