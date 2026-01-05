import numpy as np
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self):
        self.word_probs = defaultdict(lambda: {'spam': 0, 'ham': 0})
        self.class_probs = {'spam': 0.0, 'ham': 0.0}
        self.vocab = set()

    def tokenize(self, text):
        return text.lower().replace('.', '').split()

    def train(self, texts, labels):
        # 1. Count Occurrences
        counts = {'spam': 0, 'ham': 0}
        word_counts = {'spam': defaultdict(int), 'ham': defaultdict(int)}
        
        for text, label in zip(texts, labels):
            counts[label] += 1
            for word in self.tokenize(text):
                word_counts[label][word] += 1
                self.vocab.add(word)
        
        # 2. Calculate Priors P(Class)
        total_docs = len(texts)
        self.class_probs['spam'] = counts['spam'] / total_docs
        self.class_probs['ham'] = counts['ham'] / total_docs
        
        # 3. Calculate Likelihoods P(Word|Class) with Laplace Smoothing
        for word in self.vocab:
            # P(word|spam) = (count(word|spam) + 1) / (total_spam_words + vocab_size)
            # We use Log Probability to avoid underflow
            spam_denom = sum(word_counts['spam'].values()) + len(self.vocab)
            ham_denom = sum(word_counts['ham'].values()) + len(self.vocab)
            
            self.word_probs[word]['spam'] = np.log((word_counts['spam'][word] + 1) / spam_denom)
            self.word_probs[word]['ham'] = np.log((word_counts['ham'][word] + 1) / ham_denom)
            
        print("Training Complete.")

    def predict(self, text):
        # Log P(Spam|Text) ~ Log P(Spam) + Sum Log P(Word|Spam)
        score_spam = np.log(self.class_probs['spam'])
        score_ham = np.log(self.class_probs['ham'])
        
        for word in self.tokenize(text):
            if word in self.vocab:
                score_spam += self.word_probs[word]['spam']
                score_ham += self.word_probs[word]['ham']
        
        return 'spam' if score_spam > score_ham else 'ham'

if __name__ == "__main__":
    # Test Data
    X = [
        "Win a free iPhone now",
        "Meeting at 3pm today",
        "Free cash prize click here",
        "Can we reschedule the call?"
    ]
    y = ['spam', 'ham', 'spam', 'ham']
    
    clf = NaiveBayesClassifier()
    clf.train(X, y)
    
    test_msg = "Free prize today"
    print(f"Prediction for '{test_msg}': {clf.predict(test_msg)}")
