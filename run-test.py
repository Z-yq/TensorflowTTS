from utils.user_config import  UserConfig
from models.model import TTSmodel

class TTS():
    def __init__(self,acoustic_config,vocoder_config=None):

        self.tts_model=TTSmodel(acoustic_config,vocoder_config)
        self.tts_model.load_model(False)
    def tts(self,text,spk):
        wav=self.tts_model.synthesize(text,spk)
        return wav
if __name__ == '__main__':
    import argparse
    import soundfile as sf
    import time

    # Set CPU 1 core
    # import tensorflow as tf
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES']='-1'
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)
    parse=argparse.ArgumentParser()
    parse.add_argument('--common_config', type=str, default='./configs/common.yml', help='the am data config path')
    parse.add_argument('--acoustic_config', type=str, default='./configs/fastspeech.yml', help='the am model config path')
    parse.add_argument('--vocoder_config', type=str, default='./configs/vocoder.yml', help='the am model config path')
    args=parse.parse_args()
    acoustic_config=UserConfig(args.common_config,args.acoustic_config)
    vocoder_config=UserConfig(args.common_config,args.vocoder_config)
    tts=TTS(acoustic_config,vocoder_config)

    wav=tts.tts('来一句长一点的话儿试一试呢。','spk1')
    s=time.time()
    wav=tts.tts('来一句常一点的话儿试一试呢。','spk1')
    e=time.time()
    sf.write('test.wav',wav,8000)
    print('wav length:',wav.shape/8000,'tts cost time:',e-s)