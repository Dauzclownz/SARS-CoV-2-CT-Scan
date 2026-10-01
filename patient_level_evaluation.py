#!/usr/bin/env python3
"""
patient_level_evaluation.py

Reproduces Table 6 (Section 4.8) of the manuscript

    Beyond single-split accuracy: near-perfect scores on a COVID-19 CT benchmark
    do not transfer across cohorts

What the script measures
------------------------
Slice-level leakage on the larger, patient-organized multiclass CT dataset of
Soares et al. (2024), Evol Syst 15:635-640 (4,173 scans from 210 subjects in the paper; the Kaggle
release holds 4,171 slices), collapsed to
the binary task COVID versus non-COVID, where non-COVID comprises healthy lungs and other
pulmonary conditions. Dataset, task and training pipeline are held fixed and only the split
changes:

  (a) slice-level split: one stratified 80/20 split of the slices, so slices of the same
      patient can fall on both sides (the conventional protocol);
  (b) subject-disjoint cross-validation: five folds built with StratifiedGroupKFold on the
      patient identifiers, so no patient contributes slices to both training and test data.
      Fold 1 also serves as the single patient-disjoint 80/20 split discussed in Section 4.8.

Methods: xDNN on re-extracted torchvision VGG-16 features, ViT-Small, Swin-Small,
ConvNeXt-Small and their soft-voting ensemble. Inside every fold, pairwise McNemar tests are
run with a Bonferroni correction, as for Table 3.

Shared pipeline
---------------
Preprocessing, `train_deep`, `predict_probs`, `metrics`, the VGG-16 encoder and the xDNN
classifier are copied from the main notebook (Sections 2 to 4) with the same settings
(AdamW, lr 3e-5, weight decay 0.05, cosine schedule, label smoothing 0.1, class-balanced
cross-entropy, 224 x 224 inputs, ImageNet normalization, flip/rotation/color-jitter
augmentation, horizontal-flip test-time augmentation, mixed precision on CUDA, deterministic
cuDNN). The number of epochs is identical for every split, so the comparison isolates the
effect of the split.

Usage
-----
    # download the dataset with the Kaggle API (needs ~/.kaggle/kaggle.json), then run
    python patient_level_evaluation.py --download --data_dir data/multiclass

    # or point to an existing copy (Kaggle or Synapse syn22174850)
    python patient_level_evaluation.py --data_dir /path/to/multiclass

    # quick functional check on a handful of patients (minutes, not a result)
    python patient_level_evaluation.py --data_dir /path/to/multiclass \
        --subsample_patients 6 --epochs 1 --out_dir results_smoke

Patient identifiers are read from the folder layout (class folder, then one sub-folder per
patient). If a copy of the data stores images differently, pass --patient_regex to read the
patient from the file name, or --index_csv with columns path,label,patient.

The run is resumable: every finished training is checkpointed in <out_dir>/checkpoint.pkl, so
re-running the same command after a disconnect continues where it stopped.
"""

import argparse
import collections
import hashlib
import json
import math
import os
import pickle
import platform
import random
import re
import subprocess
import sys
import time
from pathlib import Path

# must be set before CUDA is initialized (deterministic cuBLAS, as in the notebook)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from statsmodels.stats.contingency_tables import mcnemar
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import VGG16_Weights, vgg16

KAGGLE_SLUG = 'plameneduardo/a-covid-multiclass-dataset-of-ct-scans'
EXPECTED = {'slices': (4171, 4173), 'patients': 210}  # 4,171 in the Kaggle release, 4,173 in Soares et al. (2024)
IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
DEFAULT_PATIENT_REGEX = r'(?i)(?:patient|paciente|pat)[\s_\-]*0*(\d+)'

MODEL_NAMES = {
    'small': {'ViT': 'vit_small_patch16_224', 'Swin': 'swin_small_patch4_window7_224',
              'ConvNeXt': 'convnext_small'},
    'base': {'ViT': 'vit_base_patch16_224', 'Swin': 'swin_base_patch4_window7_224',
             'ConvNeXt': 'convnext_base'},
}
DEEP = ['ViT', 'Swin', 'ConvNeXt']
ORDER = ['xDNN', 'ViT', 'Swin', 'ConvNeXt', 'Ensemble']
COVID = 1                                             # label convention: 1 = COVID, 0 = non-COVID

IMG = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# identical to the main notebook (Section 4)
train_tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.RandomHorizontalFlip(),
                               transforms.RandomRotation(10), transforms.ColorJitter(0.1, 0.1, 0.1),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
eval_tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(),
                              transforms.Normalize(MEAN, STD)])
feat_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(IMG),
                              transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEV_TYPE = 'cuda' if DEVICE == 'cuda' else 'cpu'


# =============================================================================================
# xDNN (Angelov P, Soares E (2020) Towards explainable deep neural networks (xDNN).
# Neural Netw 130:185-194). Same code as Section 2 of the main notebook: the original
# implementation with two compatibility fixes (Minkowski p=6 keyword, deprecated np.int removed).
# One defensive change in xDNNclassifier: X, Support and Radius start as 2-D column arrays, the
# shape the original gives them after the first new prototype. This matters only when the second
# training sample merges into the first prototype, where the original raises an IndexError; on
# every other path the arithmetic and the results are unchanged.
# =============================================================================================

def xDNN(Input, Mode):
    if Mode == 'Learning':
        Images = Input['Images']
        Features = Input['Features']
        Labels = Input['Labels']
        CN = max(Labels)
        Prototypes = PrototypesIdentification(Images, Features, Labels, CN)
        Output = {}
        Output['xDNNParms'] = {}
        Output['xDNNParms']['Parameters'] = Prototypes
        MemberLabels = {}
        for i in range(0, CN + 1):
            MemberLabels[i] = Input['Labels'][Input['Labels'] == i]
        Output['xDNNParms']['CurrentNumberofClass'] = CN + 1
        Output['xDNNParms']['OriginalNumberofClass'] = CN + 1
        Output['xDNNParms']['MemberLabels'] = MemberLabels
        return Output

    elif Mode == 'Validation':
        Params = Input['xDNNParms']
        datates = Input['Features']
        Test_Results = DecisionMaking(Params, datates)
        EstimatedLabels = Test_Results['EstimatedLabels']
        Scores = Test_Results['Scores']
        Output = {}
        Output['EstLabs'] = EstimatedLabels
        Output['Scores'] = Scores
        Output['ConfMa'] = confusion_matrix(Input['Labels'], Output['EstLabs'])
        Output['ClassAcc'] = np.sum(Output['ConfMa'] * np.identity(len(Output['ConfMa']))) / len(Input['Labels'])
        return Output


def PrototypesIdentification(Image, GlobalFeature, LABEL, CL):
    data = {}
    image = {}
    label = {}
    Prototypes = {}
    for i in range(0, CL + 1):
        seq = np.argwhere(LABEL == i)
        data[i] = GlobalFeature[seq, ]
        image[i] = {}
        for j in range(0, len(seq)):
            image[i][j] = Image[seq[j][0]]
        label[i] = np.ones((len(seq), 1)) * i
    for i in range(0, CL + 1):
        Prototypes[i] = xDNNclassifier(data[i], image[i])
    return Prototypes


def xDNNclassifier(Data, Image):
    L, N, W = np.shape(Data)
    radius = 1 - math.cos(math.pi / 6)
    data = Data.copy()
    Centre = data[0, ]
    Center_power = np.power(Centre, 2)
    X = np.array([[np.sum(Center_power)]])
    Support = np.array([[1]])
    Noc = 1
    GMean = Centre.copy()
    Radius = np.array([[radius]])
    ND = 1
    VisualPrototype = {}
    VisualPrototype[1] = Image[0]
    for i in range(2, L + 1):
        GMean = (i - 1) / i * GMean + data[i - 1, ] / i
        CentreDensity = np.sum((Centre - np.kron(np.ones((Noc, 1)), GMean)) ** 2, axis=1)
        CDmax = max(CentreDensity)
        CDmin = min(CentreDensity)
        DataDensity = np.sum((data[i - 1, ] - GMean) ** 2)
        if i == 2:
            distance = cdist(data[i - 1, ].reshape(1, -1), Centre.reshape(1, -1), 'euclidean')[0]
        else:
            distance = cdist(data[i - 1, ].reshape(1, -1), Centre, 'euclidean')[0]
        value, position = distance.max(0), distance.argmax(0)
        value = value ** 2

        if DataDensity > CDmax or DataDensity < CDmin or value > 2 * Radius[position]:
            Centre = np.vstack((Centre, data[i - 1, ]))
            Noc = Noc + 1
            VisualPrototype[Noc] = Image[i - 1]
            X = np.vstack((X, ND))
            Support = np.vstack((Support, 1))
            Radius = np.vstack((Radius, radius))
        else:
            Centre[position, ] = Centre[position, ] * (Support[position] / Support[position] + 1) + data[i - 1] / (Support[position] + 1)
            Support[position] = Support[position] + 1
            Radius[position] = 0.5 * Radius[position] + 0.5 * (X[position, ] - sum(Centre[position, ] ** 2)) / 2
    dic = {}
    dic['Noc'] = Noc
    dic['Centre'] = Centre
    dic['Support'] = Support
    dic['Radius'] = Radius
    dic['GMean'] = GMean
    dic['Prototype'] = VisualPrototype
    dic['L'] = L
    dic['X'] = X
    return dic


def DecisionMaking(Params, datates):
    PARAM = Params['Parameters']
    CurrentNC = Params['CurrentNumberofClass']
    LAB = Params['MemberLabels']
    VV = 1
    LTes = np.shape(datates)[0]
    EstimatedLabels = np.zeros((LTes))
    Scores = np.zeros((LTes, CurrentNC))
    for i in range(1, LTes + 1):
        data = datates[i - 1, ]
        R = np.zeros((VV, CurrentNC))
        Value = np.zeros((CurrentNC, 1))
        for k in range(0, CurrentNC):
            distance = np.sort(cdist(data.reshape(1, -1), PARAM[k]['Centre'], 'minkowski', p=6))[0]
            Value[k] = distance[0]
        Value = softmax(-1 * Value ** 2).T
        Scores[i - 1, ] = Value
        Value = Value[0]
        Value_new = np.sort(Value)[::-1]
        indx = np.argsort(Value)[::-1]
        EstimatedLabels[i - 1] = indx[0]
    LABEL1 = np.zeros((CurrentNC, 1))

    for i in range(0, CurrentNC):
        LABEL1[i] = np.unique(LAB[i])

    EstimatedLabels = EstimatedLabels.astype(int)
    EstimatedLabels = LABEL1[EstimatedLabels]
    dic = {}
    dic['EstimatedLabels'] = EstimatedLabels
    dic['Scores'] = Scores
    return dic


def run_xdnn(X, y, tr, va):
    """Fit xDNN on rows `tr` and return class scores (n_va x 2, column 1 = COVID) for rows `va`."""
    O1 = xDNN({'Images': np.arange(len(tr)), 'Features': X[tr], 'Labels': y[tr].astype(int)}, 'Learning')
    O2 = xDNN({'xDNNParms': O1['xDNNParms'], 'Images': np.arange(len(va)),
               'Features': X[va], 'Labels': y[va].astype(int)}, 'Validation')
    return np.array(O2['Scores'], dtype=np.float64)


# =============================================================================================
# Data indexing: one row per slice with its binary label and patient identifier
# =============================================================================================

def class_group(name):
    """Map a folder name to 'covid', 'healthy', 'others' or 'noncovid' (None if unrelated)."""
    s = re.sub(r'[\s_\-]+', '', name.lower())
    if s.startswith('non') and 'covid' in s:
        return 'noncovid'
    if 'healthy' in s or 'normal' in s:
        return 'healthy'
    if 'other' in s or 'pneumonia' in s or s == 'cap':
        return 'others'
    if 'covid' in s or 'sarscov' in s:
        return 'covid'
    return None


def find_class_root(base):
    """Shallowest directory whose immediate sub-folders include COVID and at least one non-COVID class."""
    roots = sorted((Path(r) for r, _, _ in os.walk(base)), key=lambda p: (len(p.parts), str(p)))
    for root in roots:
        dirs = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(('.', '__')))
        groups = {d: class_group(d) for d in dirs}
        found = {g for g in groups.values() if g}
        if 'covid' in found and len(found) >= 2:
            return root, {d: g for d, g in groups.items() if g}
    return None, {}


def index_from_folders(data_dir, patient_regex):
    root, classes = find_class_root(data_dir)
    if root is None:
        sys.exit(f'Could not find COVID and non-COVID class folders under {data_dir}. '
                 'Pass --index_csv with columns path,label,patient instead.')
    rx = re.compile(patient_regex) if patient_regex else None
    rows, unassigned = [], []
    for cdir, group in sorted(classes.items()):
        for p in sorted((root / cdir).rglob('*')):
            if p.suffix.lower() not in IMG_EXT or not p.is_file():
                continue
            rel = p.relative_to(root / cdir).parts
            patient = None
            if len(rel) >= 2:                              # class/<patient folder>/.../image
                patient = f'{cdir}/{rel[0]}'
            elif rx is not None:                           # class/image, patient in the file name
                m = rx.search(p.stem)
                if m:
                    patient = f'{cdir}/{m.group(1)}'
            if patient is None:
                unassigned.append(str(p))
            rows.append(dict(path=str(p), cls=cdir, group=group,
                             label=int(group == 'covid'), patient=patient))
    if unassigned:
        sys.exit(f'{len(unassigned)} images have no patient identifier (e.g. {unassigned[0]}). '
                 'Subject-disjoint splitting needs one per image: use --patient_regex to read it '
                 'from the file name, or --index_csv with columns path,label,patient.')
    return pd.DataFrame(rows)


def index_from_csv(csv_path, data_dir):
    df = pd.read_csv(csv_path)
    missing = {'path', 'label', 'patient'} - set(df.columns)
    if missing:
        sys.exit(f'--index_csv is missing columns: {sorted(missing)}')

    def to_label(v):
        s = str(v).strip().lower()
        if s in ('1', 'covid', 'covid-19', 'covid19', 'positive'):
            return 1
        if s in ('0', 'non-covid', 'noncovid', 'healthy', 'normal', 'others', 'other', 'negative'):
            return 0
        raise ValueError(f'unrecognized label {v!r}')

    def resolve(p):                                  # absolute, relative to the cwd, or relative to data_dir
        p = str(p)
        return p if os.path.isabs(p) or os.path.exists(p) else str(Path(data_dir) / p)

    df = df.copy()
    df['path'] = [resolve(p) for p in df['path']]
    missing_files = [p for p in df['path'] if not os.path.exists(p)]
    if missing_files:
        sys.exit(f'{len(missing_files)} paths in --index_csv do not exist (e.g. {missing_files[0]})')
    df['label'] = [to_label(v) for v in df['label']]
    df['patient'] = df['patient'].astype(str)
    df['cls'] = df.get('cls', pd.Series(np.where(df['label'] == 1, 'COVID', 'non-COVID')))
    df['group'] = np.where(df['label'] == 1, 'covid', 'noncovid')
    return df.sort_values('path').reset_index(drop=True)


def kaggle_download(data_dir):
    have = [p for p in Path(data_dir).rglob('*') if p.suffix.lower() in IMG_EXT] if Path(data_dir).exists() else []
    if have:
        print(f'--download: {len(have)} images already present in {data_dir}, skipping download')
        return
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    print(f'downloading {KAGGLE_SLUG} with the Kaggle API ...')
    try:
        subprocess.run(['kaggle', 'datasets', 'download', '-d', KAGGLE_SLUG, '-p', str(data_dir), '--unzip'],
                       check=True)
    except FileNotFoundError:
        sys.exit('The kaggle CLI is not installed: pip install kaggle, and place kaggle.json in ~/.kaggle/')


def subsample(df, n_per_class, seed):
    """Keep n patients per class folder; for quick functional checks only."""
    rng = np.random.RandomState(seed)
    keep = []
    for cls, g in df.groupby('cls'):
        pats = sorted(g['patient'].unique())
        keep += list(rng.choice(pats, min(n_per_class, len(pats)), replace=False))
    return df[df['patient'].isin(keep)].reset_index(drop=True)


# =============================================================================================
# Shared pipeline (copied from the main notebook, Section 4)
# =============================================================================================

class PathDataset(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = list(paths), np.asarray(labels).astype(int), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert('RGB')), int(self.labels[i])


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def set_determinism():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


class Pipeline:
    def __init__(self, paths, y, args):
        self.paths, self.y, self.args = np.asarray(paths), np.asarray(y).astype(int), args

    def loader(self, idx, tf, shuffle, batch=None):
        return DataLoader(PathDataset(self.paths[idx], self.y[idx], tf), batch_size=batch or self.args.batch_size,
                          shuffle=shuffle, num_workers=self.args.num_workers, pin_memory=(DEVICE == 'cuda'))

    def train_deep(self, model_name, tr, va, epochs, seed):
        seed_everything(seed)
        tl = self.loader(tr, train_tf, True)
        model = timm.create_model(model_name, pretrained=not self.args.no_pretrained, num_classes=2).to(DEVICE)
        cnt = collections.Counter(self.y[tr].tolist())
        w = torch.tensor([1.0 / cnt[i] for i in range(2)], dtype=torch.float); w = w / w.sum() * 2
        crit = nn.CrossEntropyLoss(weight=w.to(DEVICE), label_smoothing=0.1)
        opt = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=0.05)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == 'cuda'))
        for ep in range(epochs):
            model.train()
            for x, yb in tl:
                x, yb = x.to(DEVICE), yb.to(DEVICE); opt.zero_grad()
                with torch.amp.autocast(DEV_TYPE, enabled=(DEVICE == 'cuda')):
                    loss = crit(model(x), yb)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sch.step()
        P = self.predict_probs(model, self.loader(va, eval_tf, False))
        del model
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        return P

    @staticmethod
    def predict_probs(model, dl):
        model.eval(); out = []
        with torch.no_grad():
            for x, _ in dl:
                x = x.to(DEVICE)
                with torch.amp.autocast(DEV_TYPE, enabled=(DEVICE == 'cuda')):
                    o = model(x) + model(torch.flip(x, dims=[3]))     # horizontal-flip TTA
                out.append(torch.softmax(o.float(), 1).cpu().numpy())
        return np.concatenate(out)

    def vgg_features(self):
        enc = _VGGEnc(pretrained=not self.args.no_pretrained).eval().to(DEVICE); fe = []
        dl = self.loader(np.arange(len(self.paths)), feat_tf, False, batch=64)
        with torch.no_grad():
            for x, _ in dl:
                with torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
                    fe.append(enc(x.to(DEVICE)).float().cpu().numpy())
        del enc
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        return np.concatenate(fe).astype(np.float64)


class _VGGEnc(nn.Module):
    """VGG-16 up to the second fully connected layer (4,096-d), as in the notebook."""
    def __init__(self, pretrained=True):
        super().__init__()
        v = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.f, self.a = v.features, v.avgpool
        self.c = nn.Sequential(*list(v.classifier.children())[:-3])

    def forward(self, x):
        x = self.f(x); x = self.a(x); x = torch.flatten(x, 1); return self.c(x)


def metrics(y_true, probs):
    pred = probs.argmax(1); yp = (y_true == COVID).astype(int); pp = (pred == COVID).astype(int)
    return dict(acc=accuracy_score(y_true, pred), prec=precision_score(yp, pp, zero_division=0),
                rec=recall_score(yp, pp, zero_division=0), f1=f1_score(yp, pp, zero_division=0),
                auc=roc_auc_score(yp, probs[:, COVID]), kappa=cohen_kappa_score(y_true, pred))


def mcnemar_p(a_ok, b_ok):
    tab = [[int((a_ok & b_ok).sum()), int((a_ok & ~b_ok).sum())],
           [int((~a_ok & b_ok).sum()), int((~a_ok & ~b_ok).sum())]]
    return mcnemar(tab, exact=(tab[0][1] + tab[1][0] < 25)).pvalue


# =============================================================================================
# Main experiment
# =============================================================================================

def parse_args():
    ap = argparse.ArgumentParser(description='Table 6: slice-level versus patient-level evaluation.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--data_dir', required=True, help='root of the extracted multiclass dataset')
    ap.add_argument('--download', action='store_true', help=f'download {KAGGLE_SLUG} into --data_dir first')
    ap.add_argument('--index_csv', default=None, help='optional CSV with columns path,label,patient')
    ap.add_argument('--patient_regex', default=DEFAULT_PATIENT_REGEX,
                    help='regex with one group, used only for images stored directly in a class folder')
    ap.add_argument('--out_dir', default='results_patient_level')
    ap.add_argument('--epochs', type=int, default=8, help='training epochs, identical for every split')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--model_size', choices=['small', 'base'], default='small')
    ap.add_argument('--models', default='xdnn,vit,swin,convnext',
                    help='comma-separated subset of xdnn,vit,swin,convnext')
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--subsample_patients', type=int, default=None,
                    help='keep N patients per class folder (functional check only, not a result)')
    ap.add_argument('--no_pretrained', action='store_true',
                    help='random initialization (offline functional check only, not a result)')
    ap.add_argument('--fresh', action='store_true', help='ignore an existing checkpoint')
    return ap.parse_args()


def fingerprint(df, args):
    h = hashlib.sha256()
    h.update('\n'.join(os.path.relpath(p, args.data_dir) for p in df['path']).encode())   # location-independent
    for col in ('label', 'patient'):
        h.update('\n'.join(map(str, df[col])).encode())
    for k in ('epochs', 'folds', 'seed', 'batch_size', 'lr', 'model_size', 'subsample_patients', 'no_pretrained'):
        h.update(f'{k}={getattr(args, k)}'.encode())
    return h.hexdigest()


def fmt(v):
    return f'{v * 100:.2f}'


def main():
    args = parse_args()
    set_determinism()
    seed_everything(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    if args.download:
        kaggle_download(args.data_dir)
    df = index_from_csv(args.index_csv, args.data_dir) if args.index_csv else index_from_folders(args.data_dir, args.patient_regex)
    if args.subsample_patients:
        df = subsample(df, args.subsample_patients, args.seed)
        print(f'*** subsampled to {args.subsample_patients} patients per class: functional check, not a result ***')
    if args.no_pretrained:
        print('*** --no_pretrained: random initialization, functional check only, not a result ***')

    bad = df.groupby('patient')['label'].nunique()
    if (bad > 1).any():
        sys.exit(f'patients with mixed labels: {list(bad[bad > 1].index)[:5]}')
    y = df['label'].to_numpy().astype(int)
    groups = df['patient'].to_numpy()

    summary = (df.groupby('cls').agg(slices=('path', 'size'), patients=('patient', 'nunique'),
                                     label=('label', 'first')).reset_index())
    print('\nDataset summary (label 1 = COVID, 0 = non-COVID)')
    print(summary.to_string(index=False))
    n_sl, n_pt = len(df), df['patient'].nunique()
    print(f'total: {n_sl} slices from {n_pt} patients | COVID slices {int(y.sum())}, non-COVID {int((1 - y).sum())}')
    if not args.subsample_patients and (n_sl not in EXPECTED['slices'] or n_pt != EXPECTED['patients']):
        print(f'WARNING: expected 4,171 slices (Kaggle release; 4,173 in Soares et al. 2024) from '
              f'{EXPECTED["patients"]} patients; check that the full dataset was extracted.')

    # ------------------------------------------------------------------ splits
    idx = np.arange(len(df))
    tr_s, te_s = train_test_split(idx, test_size=0.2, stratify=y, random_state=args.seed)
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = list(sgkf.split(idx, y, groups))
    fold_of = np.empty(len(df), dtype=int)
    for f, (_, va) in enumerate(folds):
        fold_of[va] = f + 1
        assert not set(groups[folds[f][0]]) & set(groups[va]), 'patient overlap inside a fold'
    splits = {'slice': (tr_s, te_s)}
    splits.update({f'fold{f + 1}': fv for f, fv in enumerate(folds)})

    shared = set(groups[tr_s]) & set(groups[te_s])
    leak = np.isin(groups[te_s], list(shared)).mean()
    print(f'\nslice-level split: {len(shared)} of {len(set(groups[te_s]))} test patients also appear in '
          f'training ({leak * 100:.1f}% of test slices); patient-level folds share no patient.')

    df_out = df.copy()
    df_out['relpath'] = [os.path.relpath(p, args.data_dir) for p in df_out['path']]
    df_out['slice_split'] = np.where(np.isin(idx, te_s), 'test', 'train'); df_out['patient_fold'] = fold_of
    df_out.to_csv(out / 'dataset_index.csv', index=False)

    # ------------------------------------------------------------------ checkpoint
    ckpt_path = out / 'checkpoint.pkl'
    fp = fingerprint(df, args)
    state = {'fingerprint': fp, 'probs': {}}
    if ckpt_path.exists() and not args.fresh:
        old = pickle.load(open(ckpt_path, 'rb'))
        if old.get('fingerprint') != fp:
            sys.exit(f'{ckpt_path} belongs to a different dataset or configuration; '
                     'use --fresh or another --out_dir.')
        state = old
        print(f'resuming: {len(state["probs"])} finished runs found in {ckpt_path}')

    def save():
        tmp = ckpt_path.with_suffix('.tmp'); pickle.dump(state, open(tmp, 'wb')); os.replace(tmp, ckpt_path)

    wanted = [m.strip().lower() for m in args.models.split(',') if m.strip()]
    unknown = set(wanted) - {'xdnn', 'vit', 'swin', 'convnext'}
    if unknown:
        sys.exit(f'unknown --models entries: {sorted(unknown)} (choose from xdnn,vit,swin,convnext)')
    deep = [m for m in DEEP if m.lower() in wanted]
    use_xdnn = 'xdnn' in wanted
    pipe = Pipeline(df['path'].to_numpy(), y, args)
    names = MODEL_NAMES[args.model_size]

    # ------------------------------------------------------------------ xDNN on re-extracted VGG-16 features
    if use_xdnn:
        todo = [s for s in splits if ('xDNN', s) not in state['probs']]
        if todo:
            feat_path = out / 'vgg16_features.npy'
            meta_path = out / 'vgg16_features.json'
            X = None
            if feat_path.exists() and meta_path.exists() and json.load(open(meta_path)).get('fingerprint') == fp:
                X = np.load(feat_path)
            if X is None:
                t = time.time(); print('\nextracting VGG-16 features ...')
                X = pipe.vgg_features(); np.save(feat_path, X)
                json.dump({'fingerprint': fp, 'shape': list(X.shape)}, open(meta_path, 'w'))
                print(f'  features {X.shape} ({time.time() - t:.0f}s)')
            for s in todo:
                t = time.time(); tr, va = splits[s]
                state['probs'][('xDNN', s)] = run_xdnn(X, y, tr, va); save()
                print(f'xDNN {s}: done ({time.time() - t:.0f}s)')

    # ------------------------------------------------------------------ deep models
    for mi, short in enumerate(DEEP):
        if short not in deep:
            continue
        for s, (tr, va) in splits.items():
            if (short, s) in state['probs']:
                continue
            # seeds follow Table 2 of the notebook: SEED for the single split, SEED + 10*model + fold for CV
            seed = args.seed if s == 'slice' else args.seed + mi * 10 + (int(s[4:]) - 1)
            t = time.time()
            P = pipe.train_deep(names[short], tr, va, args.epochs, seed)
            state['probs'][(short, s)] = P; save()
            print(f'{short} {s}: acc {(P.argmax(1) == y[va]).mean() * 100:.2f}% ({time.time() - t:.0f}s)')

    # ------------------------------------------------------------------ ensemble and metrics
    methods = (['xDNN'] if use_xdnn else []) + deep
    if len(deep) == 3:
        for s in splits:
            state['probs'][('Ensemble', s)] = np.mean([state['probs'][(m, s)] for m in DEEP], axis=0)
        methods.append('Ensemble')
    methods = [m for m in ORDER if m in methods]

    rows = []
    for m in methods:
        for s, (tr, va) in splits.items():
            r = metrics(y[va], state['probs'][(m, s)])
            r.update(method=m, split=s, n_test=len(va), test_patients=len(set(groups[va])))
            rows.append(r)
    per = pd.DataFrame(rows)[['method', 'split', 'n_test', 'test_patients', 'acc', 'prec', 'rec', 'f1', 'auc', 'kappa']]
    per.to_csv(out / 'per_split_metrics.csv', index=False)

    fold_names = [s for s in splits if s != 'slice']
    t6 = []
    for m in methods:
        sl = per[(per.method == m) & (per.split == 'slice')].iloc[0]
        cv = per[(per.method == m) & per.split.isin(fold_names)]
        f1m, f1s = cv.f1.mean(), cv.f1.std(ddof=0)       # population SD, as in Table 2 of the notebook
        aum, aus = cv.auc.mean(), cv.auc.std(ddof=0)
        t6.append({'Method': m, 'F1 (slice)': sl.f1 * 100, 'F1 (patient, CV) mean': f1m * 100,
                   'F1 (patient, CV) sd': f1s * 100, 'dF1': (sl.f1 - f1m) * 100,
                   'AUC (slice)': sl.auc * 100, 'AUC (patient, CV) mean': aum * 100,
                   'AUC (patient, CV) sd': aus * 100, 'dAUC': (sl.auc - aum) * 100})
    t6 = pd.DataFrame(t6)
    t6.round(2).to_csv(out / 'table6.csv', index=False)

    header = '| Method | F1 (slice) | F1 (patient, CV) | ΔF1 | AUC (slice) | AUC (patient, CV) | ΔAUC |'
    lines = [header, '| --- | --- | --- | --- | --- | --- | --- |']
    for _, r in t6.iterrows():
        lines.append(f"| {r['Method']} | {r['F1 (slice)']:.2f} | {r['F1 (patient, CV) mean']:.1f} ± {r['F1 (patient, CV) sd']:.1f} "
                     f"| {r['dF1']:.1f} | {r['AUC (slice)']:.2f} | {r['AUC (patient, CV) mean']:.1f} ± "
                     f"{r['AUC (patient, CV) sd']:.1f} | {r['dAUC']:.1f} |")
    (out / 'table6.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\nTable 6: slice-level (single 80/20 split) versus patient-level (subject-disjoint CV, mean ± SD)')
    print('\n'.join(lines))

    print('\nSingle patient-disjoint split (fold 1):')
    for m in methods:
        r = per[(per.method == m) & (per.split == 'fold1')].iloc[0]
        print(f'  {m:9s} F1 {fmt(r.f1)}  AUC {fmt(r.auc)}')

    # ------------------------------------------------------------------ McNemar inside every fold
    pairs = [(a, b) for i, a in enumerate(methods) for b in methods[i + 1:]]
    alpha = 0.05 / max(len(pairs), 1)                    # Bonferroni, as in Table 3
    mc_rows = []
    for s in fold_names:
        va = splits[s][1]
        ok = {m: state['probs'][(m, s)].argmax(1) == y[va] for m in methods}
        for a, b in pairs:
            mc_rows.append(dict(comparison=f'{a} vs {b}', fold=s, p=mcnemar_p(ok[a], ok[b])))
    mc = pd.DataFrame(mc_rows)
    mc.to_csv(out / 'mcnemar_per_fold.csv', index=False)
    mcs = (mc.groupby('comparison', sort=False)
             .agg(significant_folds=('p', lambda p: int((p < alpha).sum())), median_p=('p', 'median'))
             .reset_index())
    mcs['significant_folds'] = mcs['significant_folds'].astype(str) + f'/{len(fold_names)}'
    mcs.to_csv(out / 'mcnemar_summary.csv', index=False)
    print(f'\nPairwise McNemar inside each subject-disjoint fold (Bonferroni alpha = {alpha:.4f})')
    print(mcs.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    # ------------------------------------------------------------------ fold-by-fold F1 differences
    dl = []
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            fa = per[(per.method == a) & per.split.isin(fold_names)].set_index('split').f1
            fb = per[(per.method == b) & per.split.isin(fold_names)].set_index('split').f1
            d = (fa - fb).loc[fold_names] * 100
            dl.append(dict(comparison=f'{a} - {b}', mean_dF1=d.mean(), folds_first_ahead=int((d > 0).sum()),
                           folds_tied=int((d == 0).sum()), folds=len(d)))
    dl = pd.DataFrame(dl)
    dl.round(2).to_csv(out / 'pairwise_f1_differences.csv', index=False)
    print('\nMean fold-wise F1 difference (percentage points) and folds in which the first method is ahead')
    print(dl.to_string(index=False, float_format=lambda v: f'{v:.2f}'))

    info = dict(script='patient_level_evaluation.py', date=time.strftime('%Y-%m-%d %H:%M:%S'),
                python=platform.python_version(), torch=torch.__version__, timm=timm.__version__,
                device=torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu',
                args=vars(args), slices=n_sl, patients=n_pt, fingerprint=fp,
                slice_split_shared_test_patients=len(shared), slice_split_leaked_test_fraction=float(leak))
    json.dump(info, open(out / 'run_info.json', 'w'), indent=2)
    print(f'\nresults written to {out.resolve()}')


if __name__ == '__main__':
    main()
