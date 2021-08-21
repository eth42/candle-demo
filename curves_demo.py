import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning

from candle import *
from plotting import *


def gen_wave(n_pts, std, n_periods, width=1):
	X = np.zeros((n_pts,2))
	X[:,0] = np.random.sample(n_pts)
	X[:,1] = np.sin(X[:,0] * n_periods * 2*np.pi)
	X[:,1] += np.random.normal(0,std,n_pts)
	X[:,0] *= width
	y = np.zeros(X.shape[0])
	return X,y
def add_noise(X,y,p=.1):
	Xmin = np.min(X,axis=0)
	Xmax = np.max(X,axis=0)
	Xrange = Xmax-Xmin
	Xnoise = np.random.sample((int(X.shape[0]*p),X.shape[1]))
	Xnoise = Xnoise * Xrange + Xmin
	ynoise = np.random.choice(np.unique(y),Xnoise.shape[0],replace=True)
	return np.concatenate([X,Xnoise],axis=0), np.concatenate([y,ynoise],axis=0)

def create_data():
	h = lambda v: ensure_np(*train_test_split(*v))
	return h(add_noise(
		np.concatenate([
			X
			for X,y in [
				gen_wave(600,.1,2,4),
				gen_wave(1000,.1,5,4)
			]
		]),
		np.array([*[0]*600,*[1]*1000])
	))


print("Loading data...")
X_train,y_train,X_test,y_test = create_data()

lk1,lk2 = 20, 15
mpts = np.Inf
noise_cutoff = 1
uncertainty_offset = .1

candle = CANDLE(
	lk1,lk2,
	max_pts_per_class=mpts,
	noise_cutoff=noise_cutoff,
	uncertainty_offset=uncertainty_offset
)
candle.fit(X_train,y_train)
preds = candle.predict(X_test)
print("Evaluated {:6.2f}% as noise".format(np.mean(preds==-1)*100))
print("Evaluated {:6.2f}% as uncertain".format(np.mean(preds==-2)*100))
print("Test accuracies:")
print("{:21}{:>12}{:>20}".format("Model","All data","Non-noise/undec."))
acc_print = lambda model,lpreds: print(
	"{:20}:{:>11.2f}%{:>19.2f}%".format(
		model,
		100*np.mean(y_test==lpreds),
		100*np.mean(y_test[preds>-1]==lpreds[preds>-1]),
	)
)
acc_print("CANDLE",preds)
for ref in [
		KNeighborsClassifier(candle.k2 * len(candle.cs)),
		SVC(),
		LogisticRegression(),
		MLPClassifier(max_iter=1000)
	]:
	try:
		warnings.filterwarnings("ignore", category=ConvergenceWarning)
		ref.fit(X_train,y_train)
		refpreds = ref.predict(X_test)
		acc_print(type(ref).__name__,refpreds)
	except: pass


# Create scatter plots with predictions
lX = np.concatenate([X_train,X_test],axis=0)
ly = np.concatenate([y_train,y_test],axis=0)
lX,ly = X_train,y_train
preds = candle.predict(lX)
colors = [COLORS.TOLERANCE[0],COLORS.TOLERANCE[1],COLORS.TOLERANCE[3],COLORS.TOLERANCE[-1]]
shapes = ['circle','square','diamond','cross','x-thin']
marker = lambda c: dict(
	size=6,
	color=colors[c],
	line_color=colors[c],
	symbol=shapes[c],
	line_width=(1 if shapes[c][-5:] == "-thin" else 0)
)
fig = go.Figure([
	go.Scatter(
		x=lX[ly==c][:,0],
		y=lX[ly==c][:,1],
		mode="markers",
		marker=marker(c),
		name="Class {:}".format(c)
	)
	for c in np.sort(np.unique(ly))
],layout=dict(title="True classes",yaxis=dict(scaleanchor="x")))
fig.show(renderer="browser")

fig = go.Figure([
	go.Scatter(
		x=lX[preds==c][:,0],
		y=lX[preds==c][:,1],
		mode="markers",
		marker=marker(c),
		name=
			"Class {:}".format(c)
			if c >= 0 else
			("Noise" if c == -1 else "Undecided")
	)
	for c in np.sort(np.unique(preds))
],layout=dict(title="CANDLE predictions",yaxis=dict(scaleanchor="x")))
fig.show(renderer="browser")

