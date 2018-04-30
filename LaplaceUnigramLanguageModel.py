import math, collections

class LaplaceUnigramLanguageModel:
  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.uniCount = collections.defaultdict(lambda:0)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    for sentence in corpus.corpus: # iterate over sentences in the corpus
        for datum in sentence.data: # iterate over datums in the sentence
            word = datum.word # get the word
            self.uniCount[word] = self.uniCount[word] + 1
            self.total += 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of
    the sentence using your language model. Use whatever data you computed in
    train() here.
    """
    score = 0.0
    probability = math.log(1.0/len(self.uniCount)) #using logarithm to smoothen out result.
    for word in sentence: # iterate over words in the sentence
        count = self.uniCount[word]
        score += math.log(count + 1)
        score -= probability
    # NOTE: a simpler method would be just score = sentence.size() * -
    # Math.log(words.size()).
    # we show the 'for' loop for insructive purposes.
    return score
