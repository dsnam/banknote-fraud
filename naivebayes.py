'''
A Naive Bayes classifier
'''

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors.kde import KernelDensity
seed = 11
np.random.seed(seed)

'''
use training data to calc priors
'''
def calc_prior(data,pr_class):
  df = data.iloc[list(np.where(data['class']==pr_class)[0])]
  class_prob = (float)(len(df.index))/(len(data.index))
  del df['class']
  kde = KernelDensity(kernel='gaussian',bandwidth=.1).fit(df)
  return [kde,class_prob]

'''
calc posteriors with estimated dists
'''
def calc_posterior(sample,dist):
  prob = 1
  sample = sample.iloc[:-1]
  prob *= dist[0].score(sample.values.reshape(1,-1))
  prob *= dist[1]
  return prob

df = pd.read_csv('banknote.csv')
#randomized cross validation loop
for j in xrange(10):
  mask = np.random.rand(len(df)) < .6
  df.reindex(np.random.permutation(df.index))
  train = df[mask]
  test = df[~mask]
  pos_prior = calc_prior(train,1)
  neg_prior = calc_prior(train,0)

  num_correct = 0

  for i,r in test.iterrows():
    neg_prob = calc_posterior(r,neg_prior)
    pos_prob = calc_posterior(r,pos_prior)
    choice = 1 if pos_prob > neg_prob else 0
    if choice == (int)(r['class']):
      num_correct += 1
  accuracy = (float)(num_correct)/len(test.index)
  print 'iter',j,'acc:',accuracy
