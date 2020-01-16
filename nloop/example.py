import os
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
from nloop import Text

# load the dataset
data_fname = os.path.join(".", "data", "imdb_reviews_sample.csv")

data = pd.read_csv(data_fname, error_bad_lines=False, encoding="latin")

# feed in the data into the Text class
text = Text(data, column="review")

print(f"text.n_docs = {text.n_docs}\n")

print(f"The top 10 most common words in the corpus are \n")
pprint(text.token_counter.most_common(10))

# create a wordcloud
print("Here is the wordcloud...\n")
text.show_wordcloud()
plt.show()

# and do a quick topic modeling using LDA
text.lda.run()
print("Here are some topics infered from the corpus using LDA...")
pprint(text.lda.model.show_topics(8, 4))