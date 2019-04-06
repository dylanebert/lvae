import json
import argparse

class Config():
    def __init__(self, model=None, data=None, latent_size=None):
        self.model = model
        self.data = data
        self.latent_size = latent_size

    def save(self, path):
        with open(path, 'w+') as f:
            f.write(json.dumps(self.__dict__))

    def load(self, path):
        with open(path) as f:
            self.__dict__ = json.loads(f.read())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-d', '--data', type=str, required=True)
    parser.add_argument('--latent_size', type=int, required=True)
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    config = Config(args.model, args.data, args.latent_size)
    config.save(args.path)
else:
    path = 'config/zap50k.json'
    config = Config()
    try:
        config.load(path)
    except:
        import sys
        sys.exit('Path {0} doesn\'t exist'.format(path))
