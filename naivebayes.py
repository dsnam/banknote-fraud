'''
A Naive Bayes classifier
'''

import numpy as np
import pandas as pd
from scipy.stats import norm

seed = 11
np.random.seed(seed)

'''
use training data to calc priors
'''
def calc_prior(data,pr_class):
  df = data.iloc[list(np.where(data['class']==pr_class)[0])]
  means = df.mean(axis=0)
  variances = df.var(axis=0)
  del means['class']
  del variances['class']
  return [means,variances]

'''
assume normal dist and calc posteriors
'''
def calc_posterior(sample,dist):
  prob = 1
  norms = {}
  sample = sample.iloc[:-1]
  #print sample
  for i,v in sample.iteritems():
    norms[i] = norm(loc=dist[0][i],scale=np.sqrt(dist[1][i]))

  for i,v in sample.iteritems():
    prob += np.log(norms[i].cdf(sample[i]))

  return prob

data = pd.read_csv('banknote.csv')
mask = np.random.rand(len(data)) < .8
train = data[mask]
test = data[~mask]

pos_prior = calc_prior(train,1)
neg_prior = calc_prior(train,0)

num_correct = 0
for i,r in test.iterrows():
  neg_prob = calc_posterior(r,neg_prior)
  pos_prob = calc_posterior(r,pos_prior)
  #print pos_prob,neg_prob
  choice = 1 if pos_prob > neg_prob else 0
  if choice == r['class']:
    num_correct += 1
accuracy = (float)(num_correct)/len(test.index)
print 'acc:',accuracy
