import utils


def get_files():
    files = utils.list_files(utils.get_parent_dir(__file__) + "/" + "data/midis")
    threshold = int(3 * len(files) / 4)
    train_files = files[:threshold]
    test_files = files[threshold:]
    return train_files, test_files


if __name__ == "__main__":
    train, test = get_files()
    print("There are", len(train), "train files and", len(test), "test files")
