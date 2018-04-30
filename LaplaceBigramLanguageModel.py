import math, collections

class LaplaceBigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.biCount = collections.defaultdict(lambda: 0)
    self.uniCount = collections.defaultdict(lambda: 0)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    previous = '<s>'
    for sentence in corpus.corpus:
        for datum in sentence.data:  
            token = datum.word
            bigram = (previous, token)
            self.biCount[bigram] = self.biCount[bigram] + 1
            self.uniCount[token] = self.uniCount[token] + 1
            self.total += 1
            previous = token
    pass

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    previous = '<s>'
    for token in sentence[1:]:
        bigram = (previous, token)
        count = self.biCount[bigram]
        score += math.log(count+1)
        score -= math.log(self.uniCount[previous] + len(self.biCount)) #using logarithm to smoothen out result.
        previous = token

    return score

