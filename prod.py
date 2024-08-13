from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd


def return_n_similar_qs(df, question, n):
    pipeline = Pipeline([
        ('bow', CountVectorizer(stop_words='english', lowercase=True)),
        ('tfidf', TfidfTransformer()),
    ])

    piped_matrix = pipeline.fit_transform(df.input)  # Transforming the input vectors
    query_vector = pipeline.transform([question])  # Transforming the query

    sim_matrix = cosine_similarity(query_vector, piped_matrix).flatten()

    top_vector = np.argsort(sim_matrix)[-n:][
                 ::-1]  # gets back indexes of the best n matches. ie cosine similarity descending ([::-1])
    top_matches = [(df.iloc[idx].input, sim_matrix[idx]) for idx in top_vector]

    return top_matches


def loop_trough_similarity_and_get_metric(top_matches, original):
    """
    :return: (Int) Sum of all the placements the originals made in the similarity
    """
    it_count = 0
    for match, sore in top_matches:
        if match == original:
            return it_count

        it_count += 1


df = pd.read_csv(r"C:\Users\Korisnik\NLP\ChatBot\products_weights_vocabs\insurance.csv")
insurance_corpus = df.set_index('input')['output'].to_dict()

easy_mod = [
    "Does Homeowners Insurance Cover Theft?",
    "What Makes Life Insurance Important?",
    "Does Car Insurance Decrease at Age 25?",
    "How Much Should Health Insurance Cost?",
    "Is Medicare Expected to Last 20 More Years?",
    "Is Too Many Options in Retirement Savings Plans Bad?",
    "Top Medicare Advantage Plans?",
    "Are Covered Porches Included in Home Insurance Area Calculations?",
    "Comparing Whole vs. Term Life Insurance?",
    "Steps to Cancel Nationwide Renters Insurance?",
    "Is It Possible to Opt-Out of Medicare Part A?",
    "Details on the Federal Government Retirement Plan?"
]
medium_mod = [
    "Does Homeowners Insurance Include Theft Coverage?",
    "What Are the Benefits of Having Life Insurance?",
    "Do Car Insurance Rates Decrease at Age 25?",
    "How Do You Determine a Reasonable Health Insurance Premium?",
    "Is Medicare Expected to Survive for the Next 20 Years?",
    "Is Having Too Many Options in a Retirement Savings Plan a Problem?",
    "Which Medicare Advantage Plans Are Highly Recommended?",
    "Are Covered Porches Part of Home Insurance Square Footage Calculations?",
    "Comparing Whole Life Insurance vs. Term Life Insurance: Which is Superior?",
    "Steps for Cancelling Nationwide Renters Insurance?",
    "Is It Possible to Refuse Medicare Part A Coverage?",
    "Overview of the Federal Government's Retirement Plan?"
]
hard_mod = [
    "Is stealing from me covered by Homeowners insurance",
    "What Makes Life Insurance Essential?",
    "Does Car Insurance Rates Decrease After Turning 25?",
    "What Constitutes an Affordable Health Insurance Premium?",
    "Is Medicare Expected to Persist in 20 Years?",
    "Is an Excess of Options in a Retirement Savings Plan Detrimental?",
    "Which Medicare Advantage Plans Are Highly Regarded?",
    "Are Covered Porches Included in Home Insurance Area Calculations?",
    "Which Life Insurance Option - Whole or Term - Offers Superior Benefits?",
    "Steps for Terminating Nationwide Renters Insurance?",
    "Is It Possible to Reject Medicare Part A?",
    "Overview of the Retirement Plan by the Federal Government?"
]
original_list = ['Is Theft Covered By Homeowners Insurance? ',
                 'Why Is It Important To Get Life Insurance? ',
                 'Does Auto Insurance Drop When You Turn 25? ',
                 'What Is A Good Price For Health Insurance? ',
                 'Will Medicare Be Around In 20 Years? ',
                 'Can There Be Too Much Choice In A Retirement Savings Plan? ',
                 'What Are The Best Medicare Advantage Plans? ',
                 'Do covered porches figure into the square footage of a dwelling for insurance purposes? ',
                 'Which Life Insurance Is Better Whole Or Term? ',
                 'How To Cancel Nationwide Renters Insurance? ',
                 'Can You Decline Medicare Part A? ',
                 'What Is The Federal Government Retirement Plan? ']

for mod in easy_mod:
    top_matches = return_n_similar_qs(insurance_corpus, mod, 1500)
    print(loop_trough_similarity_and_get_metric(top_matches, original_list[easy_mod.index(mod)]))
