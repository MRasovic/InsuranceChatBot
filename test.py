import pandas as pd
from itertools import islice

df = pd.read_csv(r"C:\Users\Korisnik\NLP\ChatBot\products_weights_vocabs\insurance.csv")

print(df.head())

insurance_corpus = df.set_index('input')['output'].to_dict()

for key, value in islice(insurance_corpus.items(), 5):
    print(f"{key} : {value}")