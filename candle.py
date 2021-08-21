import numpy as np
from sklearn import datasets
from scipy.spatial import cKDTree
from tqdm.notebook import tqdm


def ensure_np(*args):
	return [np.array(a) for a in args]
def sq_mahal(x,mu,icov):
	return (x-mu).T.dot(icov).dot(x-mu)
def mahal(x,mu,icov):
	return sq_mahal(x,mu,icov)**.5
def sq_euclid(x,mu,icov):
	return np.sum(np.square(x-mu))
def euclid(x,mu,icov):
	return sq_euclid(x,mu,icov)**.5
def mcov(X):
	return 1/X.shape[0] * X.T.dot(X)
def train_test_split(X,y,train_frac=.8):
	X,y = np.array(X),np.array(y)
	Xs_train = []
	ys_train = []
	Xs_test = []
	ys_test = []
	for c in np.unique(y):
		cmask = y==c
		n = np.sum(cmask)
		mask = np.random.permutation(n) <= n*train_frac
		Xs_train.append(X[cmask][mask])
		ys_train.append(y[cmask][mask])
		Xs_test.append(X[cmask][~mask])
		ys_test.append(y[cmask][~mask])
	return (
		np.concatenate(Xs_train,axis=0),
		np.concatenate(ys_train,axis=0),
		np.concatenate(Xs_test,axis=0),
		np.concatenate(ys_test,axis=0)
	)

# This helper class is used to compute the plausibility score
# for a single class via the `get_confidence` function.
# Do not instantiate this yourself but use the CANDLE class.
class CANDLEH:
	def __init__(
			self,
			X, c,
			k1, k2, eps,
			cutoff,
			max_pts,
			dist = sq_mahal,
			dist_is_sq = True
		):
		self.X = X
		self.X_red = None
		self.c = c
		self.k1 = k1
		self.k2 = k2
		self.eps = eps
		self.cutoff = cutoff
		self.max_pts = max_pts
		self.dist = dist
		self.dist_is_sq = dist_is_sq

		self.X_red = None
		self.icovs = None
		self.effective_cutoff = None
		self.effective_cutoff_mean = None
		self.effective_cutoff_std = None

		self.prepare()
	
	def prepare(self):
		if self.X.shape[0] > self.max_pts:
			self.X_red = np.random.permutation(self.X)[:self.max_pts]
		else:
			self.X_red = self.X
		tree = cKDTree(self.X)
		n, d = self.X_red.shape
		self.icovs = np.zeros((n,d,d))
		for xi, x in tqdm(
				enumerate(self.X_red),
				total=self.X_red.shape[0],
				desc="Preparing icovs for class {:}".format(self.c),
				leave=False
			):
			self.icovs[xi] = np.linalg.pinv(
				mcov(self.X[tree.query(x, self.k1+1)[1][1:]] - x)
				+ self.eps * np.eye(d)
			)
		dists = self.gen_dists(self.X_red)
		cutoff_dists = np.sort(dists,axis=1)[:,self.k2]
		if self.dist_is_sq: cutoff_dists = cutoff_dists**.5
		self.effective_cutoff_mean = np.mean(cutoff_dists)
		self.effective_cutoff_std = np.std(cutoff_dists)
		self.effective_cutoff = self.effective_cutoff_mean + self.cutoff * self.effective_cutoff_std

	def gen_dists(self, X):
		return np.array([
			[
				self.dist(x, mu, icov)
				for mu, icov in zip(self.X_red, self.icovs)
			]
			for x in tqdm(X,desc="Computing distances",leave=False)
		])
	
	def get_confidence(self, X, crop=True):
		dists = self.gen_dists(X)
		cutoff_dists = np.sort(dists,axis=1)[:,self.k2-1]
		if self.dist_is_sq: cutoff_dists = cutoff_dists**.5
		cutoff_dists = (
			(cutoff_dists - self.effective_cutoff_mean)
			/ (self.cutoff * self.effective_cutoff_std)
		)
		if crop: confidence = np.minimum(1, np.maximum(0, 1 - cutoff_dists))
		else: confidence = 1 - cutoff_dists
		return confidence
	
	def get_gradient(self,x):
		dists = self.gen_dists(x[None,:])[0]
		knn = np.argsort(dists)[:self.k2]
		weights = dists[knn] / np.sum(dists[knn])
		return np.sum([
			w * 2 * self.icovs[i].dot(self.X_red[i] - x)
			for w,i in zip(weights,knn)
		])


# The main class of this demo.
# The default distance as per equation (2) of the paper
# is the Mahalanobis distance which is used squared here to
# improve computation speed.
# By replacing the distance with the Euclidean distance,
# you can simulate the case where all covariance matrices
# are assumed to be the identity matrix.
class CANDLE:
	def __init__(
			self,
			k1=20, k2=100, eps=1e-8,
			max_pts_per_class=np.Inf,
			noise_cutoff=np.Inf,
			uncertainty_offset=0,
			dist=sq_mahal,
			dist_is_sq = True
		):
		self.k1 = k1
		self.k2 = k2
		self.X = None
		self.y = None
		self.eps = eps
		self.max_pts_per_class = max_pts_per_class
		self.noise_cutoff = noise_cutoff
		self.uncertainty_offset = uncertainty_offset
		self.dist = dist
		self.dist_is_sq = dist_is_sq

		self.cs = None
		self.class_deciders = None
	
	def __str__(self):
		return "{:}(k1={:},k2={:})".format(
			type(self).__name__,self.k1,self.k2
		)
	
	def fit(self, X, y):
		self.cs = np.sort(np.unique(y, axis=0))
		self.class_deciders = []
		for c in self.cs:
			Xc = X[y==c]
			self.class_deciders.append(CANDLEH(
				X=Xc, c=c,
				k1=self.k1, k2=self.k2, eps=self.eps,
				cutoff=self.noise_cutoff,
				max_pts=self.max_pts_per_class,
				dist = self.dist,
				dist_is_sq = self.dist_is_sq,
			))
		return self
	
	def predict(self, X):
		confidences = np.vstack([
			h.get_confidence(X)
			for h in self.class_deciders
		]).T
		predictions = self.cs[np.argmax(confidences,axis=1)]
		sorted_confidences = np.sort(confidences,axis=1)
		if len(self.class_deciders) > 1:
			predictions[np.logical_and(
				sorted_confidences[:,-1] < sorted_confidences[:,-2]+self.uncertainty_offset,
				sorted_confidences[:,-2] > 0
			)] = -2
		predictions[sorted_confidences[:,-1] == 0] = -1
		return predictions
	
	def score(self,X,y):
		return np.mean(self.predict(X)==y)
	def get_params(self,deep=True):
		return {
			'k1':self.k1,
			'k2':self.k2,
			'eps':self.eps,
			'max_pts_per_class':self.max_pts_per_class,
			'noise_cutoff':self.noise_cutoff,
			'uncertainty_offset':self.uncertainty_offset,
			'dist':self.dist,
			'dist_is_sq':self.dist_is_sq
		}
	def set_params(self, **params):
		for k,v in params.items():
			setattr(self, k, v)
		return self
	
	def get_gradient(self,X):
		confidences = np.vstack([
			h.get_confidence(X, crop=False)
			for h in self.class_deciders
		]).T
		best_classes = np.argmax(confidences,axis=1)
		gradients = np.zeros(X.shape)
		for i,x,c in zip(np.arange(X.shape[0]), X, best_classes):
			gradients[i] = self.class_deciders[c].get_gradient(x)
		return gradients


