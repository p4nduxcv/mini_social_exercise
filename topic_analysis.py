import sqlite3
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)

DATABASE = 'database.sqlite'
NUM_TOPICS = 10
WORDS_PER_TOPIC = 7


def fetch_all_text():
    print(f"Connecting to database '{DATABASE}'........")
    all_text_documents = []
    try:
        with sqlite3.connect(DATABASE) as con:
            cur = con.cursor()

            # Get all post content
            print("Fetching content from 'posts' table.......")
            for row in cur.execute("SELECT content FROM posts"):
                if row[0]:
                    all_text_documents.append(row[0])

            # Get all comment content
            print("Fetching content from 'comments' table.......")
            for row in cur.execute("SELECT content FROM comments"):
                if row[0]:
                    all_text_documents.append(row[0])

        print(
            f"Successfully fetched {len(all_text_documents)} total documents.")
    except sqlite3.Error as e:
        print(f"Database error is: {e}")
    except Exception as e:
        print(f"An error is: {e}")

    return all_text_documents


def preprocess_text(documents):

    print("Preprocessing text (tokenizing, removing stopwords).......")
    processed_docs = []
    for doc in documents:
        # Converts doc to lowercase, Tokenizes (splits into words), Removes punctuation and numbrs, deacc=True removes accents

        tokens = simple_preprocess(doc, deacc=True)

        # Remove stopwords (e.g., 'a', 'the', 'is') and short words
        final_tokens = [
            token for token in tokens
            if token not in STOPWORDS and len(token) > 2
        ]

        processed_docs.append(final_tokens)
    print("Preprocessing complete.")
    return processed_docs


def perform_LDA(processed_docs):

    # Creates the gensim dictionary and corpus, then trains the LDA model.

    if not processed_docs:
        print("No processed documents to analyze. Exiting.")
        return

    # Create Gensim Dictionary, This maps each unique word to an ID.
    print("Creating gensim dictionary...")
    dictionary = Dictionary(processed_docs)

    # Filter out extreme words
    dictionary.filter_extremes(no_below=5, no_above=0.5)

    # reate Gensim Corpus (Bag-of-Words), This converts each document into a list of (word_id, word_frequency) tuples.
    print("Creating gensim corpus (Bag-of-Words)...")
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    if not corpus:
        print("Corpus is empty after filtering. Try adjusting filter_extremes.")
        return

    # Train the LDA Model
    print(f"Training LDA model to find {NUM_TOPICS} topics........")
    # pass the corpus and ditionary, and specify the number of topics.
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        passes=10,
        random_state=100
    )

    # Print the Results
    print("\n--- 10 Most Popular Topics ---")
    print(f"Showing top {WORDS_PER_TOPIC} words for each topic:\n")

    topics = lda_model.print_topics(
        num_topics=NUM_TOPICS, num_words=WORDS_PER_TOPIC)

    for i, topic in enumerate(topics):
        print(f"Topic No.{i+1}: {topic[1]}")

    print("\nAnalysis has been completed. You must interpret these word clusters to name the topics.")


def main():
    # Get all text data from the database
    documents = fetch_all_text()

    if documents:
        # Clean and preprocess the text
        processed_docs = preprocess_text(documents)

        # Run the LDA topic modeling
        perform_LDA(processed_docs)


if __name__ == "__main__":
    main()
