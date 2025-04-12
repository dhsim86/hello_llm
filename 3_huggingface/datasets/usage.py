from datasets import load_dataset

if __name__ == '__main__':
    klue_mrc_dataset = load_dataset('klue', 'mrc')
    print(klue_mrc_dataset)
