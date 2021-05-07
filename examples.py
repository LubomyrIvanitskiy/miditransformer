

def show_first_sample_encoded():
    from preprocessing import data_preparator as dp
    data = dp.DataGenerator()
    x = next(data)[0]

    import matplotlib.pyplot as plt
    plt.imshow(x, aspect='auto')
    plt.show()

def play_first_sample():
    pass


if __name__ == '__main__':
    show_first_sample_encoded()
