import re
import math 
from pysqlite2 import dbapi2 as sqlite

def batchclassify2(classifier, filename, tag):
	fs = open(filename, 'r') 	
	for line in fs:
		print 'predict: ', 
		print classifier.classify(line),
		print  ' real: ',
		print tag

def batchtrain2(classifier, filename, tag):
        fs = open(filename, 'r') 	
	for line in fs:
		classifier.train(line, tag)

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
		self.fc = {}
		self.cc = {}
		self.thresholders = {}
		self.getfeatures = getfeatures 

	def setdb(self, dbfile):
		self.con = sqlite.connect(dbfile)
		self.con.execute("create table if not exists fc(feature, category, count)")
		self.con.execute("create table if not exists cc(category, count)")

	def incf(self, feature, cat):
		count = self.fcount(feature, cat)
		if count == 0:
			self.con.execute("insert into fc values ('%s', '%s', 1)" % (feature, cat))
		else:
			self.con.execute("update fc set count=%d where feature='%s' and category='%s'" % (count + 1, feature, cat))

	def fcount(self, feature, cat):
		res = self.con.execute("select count from fc where feature='%s' and category='%s'" % (feature, cat)).fetchone()
		if res == None: return 0
		else:	return float(res[0])

	def incc(self, cat):
		count = self.catcount(cat)
		if count == 0:
			self.con.execute("insert into cc values ('%s',1)" % (cat))
		else:
			self.con.execute("update cc set count=%d where category='%s'" % (count + 1, cat)) 
	def catcount(self, cat):
		res = self.con.execute("select count from cc where category='%s'" % (cat)).fetchone()
		if res == None: return 0
		else: return float(res[0])	

	def categories(self):
		cur = self.con.execute("select category from cc")
		return [d[0] for d in cur]

	def totalcount(self):
		res = self.con.execute("select sum(count) from cc ").fetchone();
		if res == None: return 0
		return res[0]

	def setthreshold(self, cat, threshold):
		self.thresholders[tag] = threshold 

	def getthreshold(self, cat):
		if cat not in self.thresholders:
			return 1.0
		return self.thresholders[tag]
	
	def train(self, doc, cat):
		features = self.getfeatures(doc)
		for f in features:
			self.incf(f,cat)
		self.incc(cat)
		self.con.commit()

	def fprob(self, feature, cat):
		if self.catcount(cat) == 0: return 0
		return self.fcount(feature, cat) /self.catcount(cat)

	def weightedprob(self, feature, cat, fprob, weight = 1, ap = 0.5):
		prob = fprob(feature, cat) 
		totals = sum([self.fcount(feature,c) for c in self.categories()]) 
		return (weight * ap + prob * totals) / (weight + totals)

class naivebayes(classifier):
	def docprob(self, item, cat):
		wordsList = self.getfeatures(item)
		docprob = 1.0
		for word in wordsList:
			docprob *= self.weightedprob(word, cat, self.fprob)		
		print "debug: " + str(docprob)
		return docprob

	def prob(self, item, cat):
		catprob = self.catcount(cat) * 1.0  / self.totalcount()
		return self.docprob(item, cat) * catprob

	def classify(self, item, default=None):	
		probs = {}
		max = 0.0
		for cat in self.categories():
			probs[cat] = self.prob(item, cat)
			if probs[cat] > max:
				max = probs[cat]
				best = cat
		for cat in probs:
			if cat == best:
				continue
			elif (probs[cat] * self.getthreshold(cat) > max):
				best = default	
		return best	

class fisherclassifier(classifier):
	def prob(self, feature, cat):
		clf = self.fprob(feature, cat)
		if clf == 0: return 0
		return clf / sum([self.fprob(feature, c) for c in self.categories()])

	def fisherprob(self, item, cat):
		p = 1
		features = self.getfeatures(item)
		for f in features:
			p *= self.weightedprob(f, cat, self.cprob)
		fscore = -2 * math.log(p)
		return self.invchi2(fscore, len(features)*2)

	def invchi2(self, chi, df):
		m = chi/2.0
		sum = term  = math.exp(-m)
		for i in range(1, df/2):
			term *= m/i
			sum += term 
		return min(sum,1.0)

c1=naivebayes(getwords)
c1.setdb('sample.db')
#c1=fisherclassifier(getwords)
#c1.setdb('sample.db')
#batchtrain2(c1, './1008.train', 1008)
#batchtrain2(c1, './1007.train', 1007)
batchclassify2(c1, './1007.test', 1007)
#batchclassify2(c1, './1008.test', 1008)
