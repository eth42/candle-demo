import re
import plotly.express as px
from plotly import graph_objects as go
import numpy as np
from tqdm import tqdm
def show_decision_boundaries(X,y,i=0,j=1,colors=None,classifiers=[],resolution=100,offset=1,legend=False,figargs={},fig=None,renderer=None):
	figargs["showlegend"]=legend
	figargs["yaxis"]=dict(scaleanchor="x")
	if fig is None: lfig = go.Figure(layout=figargs)
	else: lfig = fig
	legend_salt = str(np.random.randint(2**31))
	# Generate grid points.
	xs = np.linspace(np.min(X[:,i])-offset,np.max(X[:,i])+offset,resolution)
	ys = np.linspace(np.min(X[:,j])-offset,np.max(X[:,j])+offset,resolution)
	tmp_pts = np.array([[x,y] for x in xs for y in ys])
	n_classes = len(np.unique(y))
	def make_lighter(col):
		rgb = np.array([int(v) for v in re.match(r"rgb\(([0-9]+)[,\s]+([0-9]+)[,\s]+([0-9]+)\)", col).groups()])
		white = np.full(3,255,dtype=int)
		light_rgb = (white + rgb)//2
		return "rgb({:})".format(",".join(map(str,light_rgb)))
	if colors is None:
		if n_classes == 2:
			contour_colorscale = [[0,'rgb(224,237,198)'],[1,'rgb(249,222,202)']]
			pts_iter = list(zip(sorted(np.unique(y)),['rgb(132,184,25)','rgb(227,105,19)']))
		elif n_classes == 3:
			contour_colorscale = [[0,'rgb(224,237,198)'],[.5,'rgb(249,222,202)'],[1,'rgb(203,225,234)']]
			pts_iter = list(zip(range(3),['rgb(132,184,25)','rgb(227,105,19)','rgb(46,134,171)']))
		else:
			colors = px.colors.sequential.Rainbow
			colors = (colors * int(np.ceil(n_classes*1./len(colors))))[:n_classes]
			contour_colorscale = [[v,c] for v,c in zip(np.linspace(0,1,len(colors)),colors)]
			pts_iter = list(zip(sorted(np.unique(y)),[make_lighter(c) for c in colors]))
	else:
		assert len(colors) >= n_classes, "Need at least as many colors as classes"
		contour_colorscale = [
			[v,c]
			for v,c in zip(np.linspace(0,1,len(colors)),colors)
		]
		negative_offset = len(colors) - n_classes
		pts_iter = list(zip(
			[*np.arange(-negative_offset,0),*np.sort(np.unique(y))],
			[make_lighter(c) for c in colors]
		))
	# Add decision boundaries to the plot
	for classifier in tqdm(classifiers):
		classifier.fit(X[:,[i,j]],y)
		predictions = classifier.predict(tmp_pts)
		lfig.add_trace(go.Heatmap(
			x=xs,
			y=ys,
			z=predictions.reshape((len(xs),len(ys))).T,
			zsmooth=False,
			name=str(classifier),
			legendgroup=legend_salt+str(classifier),
			showlegend=legend,
			showscale=False,
			colorscale=contour_colorscale,
		))
	# Add points to the plot
	for c,col in pts_iter:
		if np.sum(y==c) == 0: continue
		lfig.add_trace(go.Scatter(
			x=X[y==c,i],
			y=X[y==c,j],
			mode="markers",
			name="Class {:}".format(c),
			marker=dict(color=col)
		))
	if fig is None:
		if renderer is None: lfig.show()
		else: lfig.show(renderer=renderer)


def show_mnist(im, title="", target=None):
    if target is None:
        layout_kwargs = dict(yaxis=dict(scaleanchor="x"))
        if title != "":
            layout_kwargs['title'] = title
        fig = go.Figure(layout=layout_kwargs)
    else:
        fig = target
    s = int((im.shape[0] / 3)**.5)
    if 3*s*s == im.shape[0]:
        is_rgb = True
    else:
        is_rgb = False
        s = int(im.shape[0]**.5)
    d = 3 if is_rgb else 1
    x = np.zeros((s,s,3))
    for i in range(3):
        j = s*s*(i % d)
        k = j + s*s
        x[:,:,i] = im[j:k].reshape((s,s))
    fig.add_trace(go.Image(
			z=x
		))
    if target is None: fig.show()




def mscatter(X,**kwargs):
	margs = dict(
		x=X[:,0],
		y=X[:,1],
		mode="markers"
	)
	for k,v in kwargs.items():
		margs[k] = v
	return go.Scatter(**margs)

def mscatter3(X,**kwargs):
	margs = dict(
		x=X[:,0],
		y=X[:,1],
		z=X[:,2],
		mode="markers",
		marker=dict(size=2,color=np.linalg.norm(X,axis=1) if not None in X else None)
	)
	for k,v in kwargs.items():
		margs[k] = v
	return go.Scatter3d(**margs)

def ellipsoid(mean, cov, n_samples=20, scale=1):
	vals,vecs = np.linalg.eig(cov)
	vecs = vecs.T
	ret = []
	for j in range(len(vecs)):
		ret.extend([
			mean
			+ scale * np.cos(a)*vecs[j]*vals[j]**.5
			+ scale * np.sin(a)*vecs[(j+1)%len(vals)]*vals[(j+1)%len(vals)]**.5
			for a in np.linspace(0,2*np.pi,n_samples+1)
		])
		ret.append([None]*mean.shape[0])
	return np.array(ret)





class COLORS:
	ORANGES=[
		"rgb(242,200,91)",
		"rgb(251,164,101)",
		"rgb(248,110,81)",
		"rgb(238,62,56)",
		"rgb(209,25,62)",
	]
	BLUES=[
		"rgb(83,204,236)",
		"rgb(25,116,211)",
		"rgb(0,1,129)",
	]
	GREENS=[
		"rgb(204,255,204)",
		"rgb(179,230,185)",
		"rgb(153,204,166)",
		"rgb(128,179,147)",
		"rgb(102,153,128)",
		"rgb(77,128,108)",
		"rgb(51,102,89)",
		"rgb(26,77,70)",
		"rgb(0,51,51)",
	]
	TOLERANCE=[
		"rgb(51,34,136)",
		"rgb(17,119,51)",
		"rgb(68,170,153)",
		"rgb(136,204,238)",
		"rgb(221,204,119)",
		"rgb(204,102,119)",
		"rgb(170,68,153)",
		"rgb(136,34,85)",
	]
	SHOWCASE=[
		"rgb(51,34,136)",
		"rgb(115,199,185)",
		"rgb(204,102,119)",
	]

	def to_scale(colors):
		return [
			[c,v]
			for c,v in zip(c,np.linspace(0,1,len(colors)))
		]




