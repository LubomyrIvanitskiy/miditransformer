import midiwrap

from preprocessing import dataset_loader
from preprocessing import note_encoder as ne

emb_dim = 32


class DataGenerator:

    def __init__(self):
        self.train, self.test = dataset_loader.get_files()
        self.generator = self._data_generator()

    def _data_generator(self):
        for i, f in enumerate(self.test):
            x = ne.encode_notes(midiwrap.MidiFile(f), emb_dim=emb_dim)
            yield x, f, i

    def __next__(self):
        return next(self.generator)

    def __iter__(self):
        return self


if __name__ == '__main__':
    datagen = DataGenerator()
    print(next(datagen)[0].shape)
