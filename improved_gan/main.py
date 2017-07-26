from utils.config import Config
import trainer as t


def main():
    config = Config()
    trainer = t.Trainer(config)
    trainer.fit()

if __name__ == '__main__':
    main()