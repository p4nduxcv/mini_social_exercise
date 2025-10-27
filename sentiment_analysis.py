import sqlite3
import re
import warnings
import ssl  # For fixing download errors
import nltk
from collections import defaultdict

# NLTK imports
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Gensim imports
from gensim.corpora import Dictionary
from gensim.models import LdaModel

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
DATABASE = 'database.sqlite'
NUM_TOPICS = 10
WORDS_PER_TOPIC = 7

# --- 1. NLTK Data Download (with SSL Fix) ---


def download_nltk_data():
    """
    Downloads all necessary NLTK data models.
    Includes a fix for macOS SSL certificate errors.
    """
    print("Verifying and downloading NLTK data...")
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    packages = ['vader_lexicon', 'punkt', 'stopwords', 'wordnet']
    for package in packages:
        try:
            nltk.download(package, quiet=True)
        except Exception as e:
            print(f"Could not download {package}. Error: {e}")

    try:
        # Final check for VADER, which is critical
        nltk.data.load(
            'sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt')
        print("All NLTK data is ready.")
    except LookupError:
        print("="*50)
        print("FATAL ERROR: VADER lexicon not found.")
        print("Please run the manual NLTK download steps in your terminal.")
        print("="*50)
        raise

# --- 2. Data Extraction ---


def fetch_all_text():
    """Extracts all post and comment content from the database."""
    print(f"Connecting to database '{DATABASE}'...")
    all_text_documents = []
    try:
        with sqlite3.connect(DATABASE) as con:
            cur = con.cursor()
            print("Fetching content from 'posts' and 'comments' tables...")

            # Fetch from posts
            for row in cur.execute("SELECT content FROM posts WHERE content IS NOT NULL AND content != ''"):
                all_text_documents.append(row[0])
            # Fetch from comments
            for row in cur.execute("SELECT content FROM comments WHERE content IS NOT NULL AND content != ''"):
                all_text_documents.append(row[0])

        print(
            f"Successfully fetched {len(all_text_documents)} total documents.")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    return all_text_documents

# --- 3. Sentiment Analysis (VADER) ---


def get_sentiment_scores(raw_texts):
    """
    Analyzes a list of RAW text documents and returns a list of
    VADER compound sentiment scores.
    """
    print("Initializing VADER Sentiment Analyzer...")
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = []

    print(f"Analyzing sentiment for {len(raw_texts)} documents...")
    for text in raw_texts:
        score = sid.polarity_scores(text)['compound']
        sentiment_scores.append(score)

    print("Sentiment analysis complete.")
    return sentiment_scores

# --- 4. Topic Modeling (Gensim) ---


def preprocess_text_for_lda(texts):
    """
    Preprocesses text for LDA: tokenizes, removes stopwords, lemmatizes.
    This is different from VADER, which needs raw text.
    """
    print("Preprocessing text for LDA...")
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    processed_texts = []
    # Keep track of which documents were successfully processed
    original_indices = []

    for i, text in enumerate(texts):
        # 1. Clean (remove URLs, special chars, digits)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()

        # 2. Tokenize
        try:
            tokens = word_tokenize(text)
        except LookupError:
            # This happens if 'punkt' is still missing
            print(
                f"Warning: Skipping document {i} due to missing 'punkt' tokenizer.")
            continue

        # 3. Lemmatize and remove stopwords
        processed_tokens = [
            lemmatizer.lemmatize(token) for token in tokens
            if token not in stop_words and len(token) > 2
        ]

        if processed_tokens:  # Only add if not empty
            processed_texts.append(processed_tokens)
            original_indices.append(i)  # Save the index of this doc

    print(
        f"LDA preprocessing complete. {len(processed_texts)} valid documents for modeling.")
    return processed_texts, original_indices


def perform_lda_analysis(processed_texts, num_topics=10):
    """Performs LDA topic modeling"""
    print("Creating gensim dictionary and corpus...")
    dictionary = Dictionary(processed_texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)

    if not dictionary:
        print("Dictionary is empty after filtering. Not enough data for LDA.")
        return None, None, None

    corpus = [dictionary.doc_bow(text) for text in processed_texts]

    print(f"Training LDA model to find {num_topics} topics...")
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=100,
        passes=10
    )
    print("LDA model training complete.")
    return lda_model, corpus, dictionary

# --- 5. Linking and Calculation ---


def map_sentiment_to_topics(lda_model, corpus, sentiment_scores, original_indices):
    """
    Links each document's dominant topic to its sentiment score.
    Returns a dictionary of: {topic_id: [list_of_scores]}
    """
    print("Mapping sentiment scores to dominant topics...")
    topic_sentiment_map = defaultdict(list)

    if not corpus:
        print("Warning: LDA corpus is empty. Cannot map topics.")
        return topic_sentiment_map

    for i, doc_corpus in enumerate(corpus):
        # 'i' is the index in the *processed* list
        # 'original_doc_index' is the index in the *original* raw_texts list
        original_doc_index = original_indices[i]

        # Get the list of topics for this doc
        doc_topics = lda_model.get_document_topics(
            doc_corpus, minimum_probability=0.1)

        if doc_topics:
            # Find the topic with the highest probability
            dominant_topic = max(doc_topics, key=lambda x: x[1])
            topic_id = dominant_topic[0]

            # Get the sentiment score for this *same* document
            sentiment = sentiment_scores[original_doc_index]

            # Append the score to that topic's list
            topic_sentiment_map[topic_id].append(sentiment)

    print("Mapping complete.")
    return topic_sentiment_map


def classify_sentiment(score):
    """Classifies a compound score into a human-readable string."""
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# --- 6. Main Execution ---


def main():
    # 1. Setup
    download_nltk_data()

    # 2. Get all raw text data
    raw_texts = fetch_all_text()
    if not raw_texts:
        print("No text data found. Exiting.")
        return

    # 3. Get sentiment for all raw text
    # We do this FIRST, as VADER needs the raw, unprocessed text
    sentiment_scores = get_sentiment_scores(raw_texts)

    # 4. Get topics
    # Now we process the text for LDA
    processed_texts, original_indices = preprocess_text_for_lda(raw_texts)

    # We must filter the sentiment_scores list to match the processed_texts
    # This is NOT needed if we use original_indices in the map function
    # filtered_sentiment_scores = [sentiment_scores[i] for i in original_indices]

    lda_model, corpus, dictionary = perform_lda_analysis(
        processed_texts, num_topics=NUM_TOPICS
    )

    # --- Answer 1: Overall Platform Tone ---
    overall_avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    overall_tone = classify_sentiment(overall_avg_sentiment)

    print("\n" + "="*60)
    print("ANSWER 1: OVERALL PLATFORM TONE")
    print("="*60)
    print(f"Total Documents Analyzed: {len(sentiment_scores)}")
    print(f"Average Sentiment Score: {overall_avg_sentiment:.4f}")
    print(f"Overall Platform Tone: {overall_tone}")
    print("(Scores range from -1.0 (Negative) to +1.0 (Positive))")

    if not lda_model:
        print("\n" + "="*60)
        print("LDA model training failed (not enough data).")
        print("Cannot calculate sentiment by topic.")
        print("="*60)
        return

    # --- Answer 2: Sentiment by Topic ---
    topic_sentiment_map = map_sentiment_to_topics(
        lda_model, corpus, sentiment_scores, original_indices
    )

    # Get the topic keywords for display
    topic_keywords = {}
    all_topics = lda_model.print_topics(
        num_topics=NUM_TOPICS, num_words=WORDS_PER_TOPIC)
    for topic_id, topic_str in all_topics:
        keywords = [word.split('*')[1].replace('"', "").strip()
                    for word in topic_str.split(' + ')]
        topic_keywords[topic_id] = ", ".join(keywords)

    # Calculate average sentiment for each topic
    topic_results = []
    for topic_id, scores in topic_sentiment_map.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            topic_results.append({
                "id": topic_id,
                "avg_sentiment": avg_score,
                "tone": classify_sentiment(avg_score),
                "doc_count": len(scores),
                "keywords": topic_keywords.get(topic_id, "N/A")
            })

    # Sort results by sentiment (most positive to most negative)
    topic_results.sort(key=lambda x: x['avg_sentiment'], reverse=True)

    print("\n" + "="*60)
    print("ANSWER 2: SENTIMENT BY TOPIC")
    print("="*60)

    for topic in topic_results:
        print(
            f"\nTopic #{topic['id']} (Avg. Sentiment: {topic['avg_sentiment']:.3f} - {topic['tone']})")
        print(f"   Keywords: {topic['keywords']}")
        print(f"   (Based on {topic['doc_count']} documents)")


if __name__ == "__main__":
    main()
