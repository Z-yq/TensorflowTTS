from models.model import AMmodel
from trainer import vocoder_trainer
from dataloaders import vocoder_dataloader
from utils.user_config import UserConfig
import tensorflow as tf
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
gpus = tf.config.experimental.list_physical_devices('GPU')
logging.info('valid gpus:%d' % len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class Trainer():
    def __init__(self,config):
        self.config=config
        self.am_model=AMmodel(vocoder_config=config)
        self.am_model.load_model(True)
        self.dg=vocoder_dataloader.VocoderDataLoader(self.config)
        self.trainer = vocoder_trainer.VocoderTrainer(self.config)
        self.opt = tf.keras.optimizers.Adamax(lr=self.config['learning_rate'], beta_1=self.config['beta_1'],
                                              beta_2=self.config['beta_2'],
                                              epsilon=self.config['epsilon'])
        if self.config['use_gan']:
            self.model_d=self.am_model.discriminator
            self.opt2 = tf.keras.optimizers.Adamax(lr=self.config['learning_rate'], beta_1=self.config['beta_1'],
                                                  beta_2=self.config['beta_2'],
                                                  epsilon=self.config['epsilon'])
        else:
            self.model_d=None
            self.opt2=None
        all_train_step = self.dg.get_per_epoch_steps() * self.config['num_epochs']
        self.trainer.set_total_train_steps(all_train_step)
        self.trainer.compile(self.am_model.vocoder, self.opt,self.model_d,self.opt2)
        self.dg.batch=self.trainer.global_batch_size

    def run(self,):
        train_datasets = tf.data.Dataset.from_generator(self.dg.generator,
                                                        self.dg.return_data_types(),
                                                        self.dg.return_data_shape(),
                                                        args=(True,))
        eval_datasets = tf.data.Dataset.from_generator(self.dg.generator,
                                                       self.dg.return_data_types(),
                                                       self.dg.return_data_shape(),
                                                       args=(False,))
        self.trainer.set_datasets(train_datasets, eval_datasets)
        while 1:
            self.trainer.fit(epoch=self.dg.epochs)
            if self.trainer._finished():
                self.trainer.save_checkpoint()
                logging.info('Finish training!')
                break
            if self.trainer.steps%self.config['save_interval_steps']==0:
                self.dg.save_state(self.config['outdir'])
if __name__ == '__main__':
    import argparse
    parse=argparse.ArgumentParser()
    parse.add_argument('--data_config', type=str, default='./configs/common.yml', help='the am data config path')
    parse.add_argument('--model_config', type=str, default='./configs/vocoder.yml', help='the am model config path')
    args=parse.parse_args()

    config=UserConfig(args.data_config,args.model_config)
    train=Trainer(config)
    train.run()