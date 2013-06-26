import re
import math 

def batchtrain(classifier):
	classifier.train('No body owns the water', 'good')
	classifier.train('the quick rabbit jumps fences', 'good')
	classifier.train('buy pharmaceuticals now', 'bad')
	classifier.train('make quick money at the online casino', 'bad')
	classifier.train('the quick brown fox jumps', 'good')

def getwords(doc):
	pattern = re.compile('\\W+')
	wordList = [word.lower() for word in pattern.split(doc) if len(word) > 2 and len(word) < 20]
	return dict([(word,1) for word in wordList]) 

class classifier:
	def __init__(self, getfeatures): 
		self.catDocCount = {}
		self.featureCatDocCount = {}
		self.thresholds = {}
		self.getfeatures = getfeatures 

	def setthreshold(self, cat, threshold):
		self.thresholds[tag] = threshold 

	def getthreshold(self, cat):
		if cat not in self.thresholds:
			return 1.0
		return self.thresholds[tag]
	
	def train(self, doc, cat):
		featureDict = self.getfeatures(doc)

		self.catDocCount.setdefault(cat, 0)
		self.catDocCount[cat] += 1

		for feature in featureDict:
			self.featureCatDocCount.setdefault(feature, {})
			self.featureCatDocCount[feature].setdefault(cat, 0)
			self.featureCatDocCount[feature][cat] += 1

	def fprob(self, feature, cat):
		if feature not in self.featureCatDocCount.keys():
			return 0.0 
		count = 0	
		if cat in self.featureCatDocCount[feature]:
			count = self.featureCatDocCount[feature][cat] * 1.0	
		return  count  / self.catDocCount[cat]

	def weightedprob(self, feature, cat, fprob, weight = 1, ap = 0.5):
		prob = fprob(feature, cat) 
		totals = 0 
		for cat in self.catDocCount:
			if cat in self.featureCatDocCount[feature]:
				totals += self.featureCatDocCount[feature][cat]
		return (weight * ap + prob * totals) / (weight + totals)

class naivebayes(classifier):
	def docprob(self, item, cat):
		wordsList = self.getfeatures(item)
		docprob = 1.0
		for word in wordsList:
			docprob *= self.weightedprob(word, cat, self.fprob)		
		return docprob

	def prob(self, item, cat):
		catprob = self.catDocCount[cat] * 1.0  / sum(self.catDocCount.values())
		return self.docprob(item, cat) * catprob

	def classify(self, item, default=None):	
		probs = {}
		max = 0.0
		for cat in self.catDocCount.keys():
			probs[cat] = self.prob(item, cat)
			if probs[cat] > max:
				max = probs[cat]
				best = cat
		for cat in probs.keys():
			if cat == best:
				continue
			elif (probs[cat] * self.getthreshold(cat) > max):
				best = default	
		return best	

