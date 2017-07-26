from utils.config import Config
import trainer_dcgan as t



def main():
    config = Config(checkpoint_basename='./dcgan')
    trainer = t.TrainerDCGAN(config)
    trainer.fit()

if __name__ == '__main__':
    main()