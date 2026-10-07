"""Auxiliary experiments on ground-truth object crops, not end-to-end detection.
Train fits feature scaler, PCA, classifiers and clusters; validation only transforms.
"""
import argparse
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mpl'))
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def features(split, limit, seed):
    paths=(ROOT/f'data/prepared/{split}.txt').read_text(encoding='utf-8').splitlines()
    rng=np.random.default_rng(seed);rng.shuffle(paths)
    X,y=[],[]
    for path in paths[:limit]:
        path=Path(path)
        with Image.open(path) as image:
            image=image.convert('RGB');width,height=image.size
            for line in (path.parent.parent/'labels'/f'{path.stem}.txt').read_text().splitlines():
                c,x,cy,w,h=map(float,line.split())
                bounds=(max(0,int((x-w/2)*width)),max(0,int((cy-h/2)*height)),min(width,int((x+w/2)*width)),min(height,int((cy+h/2)*height)))
                if bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:continue
                crop=np.asarray(image.crop(bounds).resize((16,16)),dtype=np.float32)/255
                # Tiny RGB grid + color histograms: interpretable, no hidden pretrained model.
                hist=np.concatenate([np.histogram(crop[...,ch],bins=8,range=(0,1),density=False)[0]/256 for ch in range(3)])
                X.append(np.concatenate([crop.ravel(),hist]));y.append(int(c))
    if len(X)<3:raise ValueError(f'Too few annotated crops in {split}')
    return np.asarray(X),np.asarray(y)

def main(args):
    train,y=features('train',args.images,42)
    val,yv=features('val',args.images,43)
    if len(np.unique(y))<2:raise ValueError('Need at least two train classes')
    scale=StandardScaler().fit(train)
    train=scale.transform(train);val=scale.transform(val)
    pca=PCA(n_components=min(32,len(train)-1,train.shape[1]),random_state=42).fit(train)
    z=pca.transform(train);zv=pca.transform(val)
    results={}
    out=ROOT/'artifacts/features';out.mkdir(parents=True,exist_ok=True)
    for name,model in [('logistic_regression',LogisticRegression(max_iter=1000,random_state=42)),('gaussian_naive_bayes',GaussianNB())]:
        model.fit(z,y);pred=model.predict(zv)
        results[name]={'accuracy':float(accuracy_score(yv,pred)),'macro_f1':float(f1_score(yv,pred,zero_division=0)),
                       'confusion_matrix':confusion_matrix(yv,pred).tolist()}
    k=min(8,len(z)-1)
    cluster=KMeans(n_clusters=k,n_init=10,random_state=42).fit(z)
    unique=len(np.unique(cluster.labels_))
    silhouette=float(silhouette_score(z,cluster.labels_,sample_size=min(2000,len(z)),random_state=42)) if 1<unique<len(z) else None
    fig,ax=plt.subplots(figsize=(8,6))
    ax.scatter(z[:,0],z[:,1] if z.shape[1]>1 else np.zeros(len(z)),c=cluster.labels_,s=8,cmap='tab10',alpha=.6)
    ax.set(xlabel='Train PCA component 1',ylabel='Train PCA component 2',title='Train crop features: K-means clusters')
    fig.tight_layout();fig.savefig(out/'pca_clusters.png',dpi=150);plt.close(fig)
    result={'task':'ground-truth crop classification; NOT detection mAP','seed':42,'max_images_per_split':args.images,
            'train_crops':len(y),'validation_crops':len(yv),'train_class_count':len(np.unique(y)),
            'validation_classes_missing_from_train':sorted(set(yv.tolist())-set(y.tolist())),
            'pca_dimensions':z.shape[1],'pca_explained_variance':float(pca.explained_variance_ratio_.sum()),
            'kmeans_clusters':k,'train_silhouette':silhouette,'classification':results,
            'bayes_note':'GaussianNB estimates P(class|features) via Bayes rule with Gaussian conditional likelihoods and empirical class priors. This is not a Bayesian neural network or YOLO calibration.',
            'limitations':'Small random image sample; correlated crops; handcrafted RGB features; ground-truth boxes supplied; validation only, no final test access.'}
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--images',type=int,default=1000)
    main(p.parse_args())
