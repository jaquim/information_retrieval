"""
Implementation of Position-Based Click model.
Training is done with the EM algorithm.
"""


from collections import Counter, defaultdict
from decimal import Decimal
import pandas as pd
import random


class PositionBasedModel:

    def __init__(self):
        self.gamma = defaultdict(Decimal)
        self.alpha = defaultdict(lambda: defaultdict(Decimal))

    def train(self, training_file = "YandexRelPredChallenge.txt", iterations=10):

        # Read data into dataframe
        columns = ["SessionID", "TimePassed", "TypeOfAction", "TargetID", "RegionID", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        df = pd.read_csv(training_file, sep='\t', header=None, names=columns)
        print("Training with EM...")

        for i in range(iterations):

            # Initialise sums and counts
            gamma_count = defaultdict(Decimal)
            alpha_count = defaultdict(lambda: defaultdict(Decimal))

            gamma_sum = defaultdict(Decimal)
            alpha_sum = defaultdict(lambda: defaultdict(Decimal))

            # Iterate sessions
            grouped = df.groupby("SessionID")
            for session_id, session_df in grouped:

                # Extract session clicks
                session_clicks = session_df[session_df["TypeOfAction"] == "C"]["TargetID"].tolist()
                # print(session_clicks)

                # Iterate session queries
                for index, row in session_df[session_df["TypeOfAction"] == "Q"].tail(1).iterrows():
                    query_id = row["TargetID"]
                    for rank in range(1, 11): # from rank 1 to 10
                        document_id = row[rank]

                        # Determine what values should be added to the EM formula sums
                        if document_id in session_clicks:
                            gamma_value = alpha_value = 1
                        else:
                            gamma_value = (self.gamma[rank] * (1 - self.alpha[document_id][query_id])) / \
                                          (1 - self.gamma[rank] * self.alpha[document_id][query_id])
                            alpha_value = ((1 - self.gamma[rank]) * self.alpha[document_id][query_id]) / \
                                          (1 - self.gamma[rank] * self.alpha[document_id][query_id])

                        gamma_sum[rank] += gamma_value
                        alpha_sum[document_id][query_id] += alpha_value

                        gamma_count[rank] += 1
                        alpha_count[document_id][query_id] += 1

            # Update variables
            for rank, param in self.gamma.items():
                self.gamma[rank] = gamma_sum[rank] / gamma_count[rank]

            for document_id, document_params in self.alpha.items():
                for query_id, param in document_params.items():
                    self.alpha[document_id][query_id] = alpha_sum[document_id][query_id] / alpha_count[document_id][query_id]

            print("Completed iteration", i+1)
        print("Training complete")

    def click_prob(self, epsilon, rank, relevance):
        # Calculate the probability of clicking on a document
        attract = epsilon if relevance == 0 else 1-epsilon
        click_prob = float(self.gamma[rank]) * attract
        return click_prob

    def click_doc(self, epsilon, rank, relevance):
        # Decide whether a document is clicked on
        random_number = random.uniform(0, 1)
        return True if random_number < self.click_prob(epsilon, rank, relevance) else False

