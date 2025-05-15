import os
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost
import matplotlib.pyplot as plt
import mplhep as hep

# variables abbreviations:
# ld = lead, sld = sublead, M = diphoton invariant mass, id = MVA ID, Pho = photon, Lep = lepton,

var = [
    "ldPho_pt_to_M",                  # f0
    "sldPho_pt_to_M",                 # f1
    "ldPho_eta",
    "sldPho_eta",
    "cos_delta_phi_ldPho_sldPho",
    "max_Pho_id",                     # f5
    "min_Pho_id",
    "ldPho_pixel_seed",
    "sldPho_pixel_seed",
    "ld_Lep_pt",
    "ld_Lep_eta",                     # f10
    "delta_R_ldPho_ldLep",
    "delta_R_sldPho_ldLep",
    "MET_pt",
    "MET_SumEt",
    "MET_Lep_mt",                     # f15
    "Jet_multiplicity",
    "Jet0_pt",
    "Jet1_pt",
    "Jet2_pt",
    "Jet3_pt",                        # f20
    "jet_max_btagPNetB",
    "jet_max_btagDeepFlavB",
    "jet_max_btagRobustParTAK4B",
    "delta_phi_diPho_W_Lep",
    "min_delta_phi_W_Jets",           # f25
    "WH_Lep_pt_balance"
]

outdir = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT"

####################################################################################
###### Background npy for training and validation ######
####################################################################################
##### Diphoton #####
Diphoton_train_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/Diphoton_train_val.npy"
Diphoton_val_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/Diphoton_train_val.npy"
##### Drell-Yan jets #####
DYJets_train_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/DYJets_train_val.npy"
DYJets_val_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/DYJets_train_val.npy"
##### gamma + jets #####
GJets_train_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/GJets_train_val.npy"
GJets_val_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/GJets_train_val.npy"
##### QCD has no surviving event, it is therefore not included here. #####
##### Top quarks #####
Top_train_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/Top_train_val.npy"
Top_val_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/Top_train_val.npy"
##### VV #####
VV_train_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/VV_train_val.npy"
VV_val_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/VV_train_val.npy"
##### W + gamma #####
WG_train_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/WG_train_val.npy"
WG_val_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/WG_train_val.npy"
##### Z + gamma #####
ZG_train_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/ZG_train_val.npy"
ZG_val_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/ZG_train_val.npy"
####################################################################################
##### ggH #####
ggH_train_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/ggH_train.npy"
ggH_val_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/ggH_val.npy"
##### VBF #####
VBF_train_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/VBF_train_val.npy"
VBF_val_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/VBF_train_val.npy"
##### VH-hadronic (W -> 2Q or Z -> 2Q) #####
VH_bkg_train_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/VH_bkg_train_val.npy"
VH_bkg_val_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/VH_bkg_train_val.npy"
##### ZH-leptonic (Z -> 2L or Z -> LNu) #####
ZH_Zto2L_signal_train_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/ZH_Zto2L_signal_train_val.npy"
ZH_Zto2L_signal_val_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/ZH_Zto2L_signal_train_val.npy"
ZH_Zto2Nu_signal_train_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/ZH_Zto2Nu_signal_train_val.npy"
ZH_Zto2Nu_signal_val_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/ZH_Zto2Nu_signal_train_val.npy"
##### ttH #####
ttH_train_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/ttH_train_val.npy"
ttH_val_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/ttH_train_val.npy"
####################################################################################
###### Signal npy for training and validation ######
####################################################################################
WH_signal_train_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/WH_signal_train_val.npy"
WH_signal_val_path = "/eos/home-s/shuofu/my_higgsdna/VH_to_leptonic_GG/WH_BDT/WH_signal_train_val.npy"
####################################################################################
###### Training and test sample preparation ######
####################################################################################
# # Load full WH signal dataset
# WH_signal_full = np.load(WH_signal_train_path)[ : , : ]

# np.random.seed(42)  # for reproducibility
# np.random.shuffle(WH_signal_full)

# # Split 70/30
# split_idx = int(0.7 * len(WH_signal_full))
# X_sig_train = WH_signal_full[ : split_idx]
# X_sig_val   = WH_signal_full[split_idx : ]

# X_sig_train = np.load(WH_signal_train_path)[ : , : ]

# X_bkg_train = np.concatenate([
#     np.load(Diphoton_train_path)[ : , : ],
#     np.load(DYJets_train_path)[ : , : ],
#     np.load(GJets_train_path)[ : , : ],
#     np.load(Top_train_path)[ : , : ],
#     np.load(VV_train_path)[ : , : ],
#     np.load(WG_train_path)[ : , : ],
#     np.load(ZG_train_path)[ : , : ],
#     np.load(VH_bkg_train_path)[ : , : ],
#     np.load(ttH_train_path)[ : , : ],
#     np.load(ggH_train_path)[ : , : ]
# ])
# X_bkg_val = np.concatenate([
#     np.load(Diphoton_val_path)[ : , : ],
#     np.load(DYJets_val_path)[ : , : ],
#     np.load(GJets_val_path)[ : , : ],
#     np.load(Top_val_path)[ : , : ],
#     np.load(VV_val_path)[ : , : ],
#     np.load(WG_val_path)[ : , : ],
#     np.load(ZG_val_path)[ : , : ],
#     np.load(VH_bkg_val_path)[ : , : ],
#     np.load(ttH_val_path)[ : , : ],
#     np.load(ggH_val_path)[ : , : ]
# ])

# # Labels: 1 for signal, 0 for background
# y_sig_train = np.ones(len(X_sig_train))
# y_bkg_train = np.zeros(len(X_bkg_train))
# y_sig_val   = np.ones(len(X_sig_val))
# y_bkg_val   = np.zeros(len(X_bkg_val))

# # Concatenate data and labels
# X_train = np.concatenate((X_sig_train, X_bkg_train), axis=0)
# X_val   = np.concatenate((X_sig_val,   X_bkg_val), axis=0)

# y_train = np.concatenate((y_sig_train, y_bkg_train), axis=0)
# y_val   = np.concatenate((y_sig_val,   y_bkg_val), axis=0)

X_sig_train = np.load(WH_signal_train_path)[ : , : ]

X_bkg_train = np.concatenate([
    np.load(Diphoton_train_path)[ : , : ],
    np.load(DYJets_train_path)[ : , : ],
    np.load(GJets_train_path)[ : , : ],
    np.load(Top_train_path)[ : , : ],
    np.load(VV_train_path)[ : , : ],
    np.load(WG_train_path)[ : , : ],
    np.load(ZG_train_path)[ : , : ],
    np.load(VH_bkg_train_path)[ : , : ],
    np.load(ttH_train_path)[ : , : ],
    np.load(ggH_train_path)[ : , : ]
])

# Combine signal and background datasets
x = np.concatenate((X_sig_train, X_bkg_train), axis=0)
y = np.concatenate((np.ones(len(X_sig_train)), np.zeros(len(X_bkg_train))), axis=0)

# Shuffle and split into 70% train and 30% validation
X_train, X_val, Y_train, Y_val = train_test_split(
    x, y, test_size=0.3, random_state=69, stratify=y
)

####################################################################################
###### XGBoost training engine ######
####################################################################################
# XGBoost sklearn configuration
n_estimators = 1500
max_depth = 3
learning_rate = 0.05
early_stopping_rounds = 10

XGBEngine = xgboost.XGBClassifier(
    n_estimators  = n_estimators,
    learning_rate = learning_rate,
    max_depth     = max_depth,
    tree_method   = "hist",
    device        = "cuda",  # just say "cuda", not "cuda:0" for sklearn API
    eval_metric   = ["logloss"],  # can be a string if single metric
    early_stopping_rounds = early_stopping_rounds,
    # use_label_encoder=False  # recommended for sklearn >= 1.0
)

# Make sure labels are consistent with the variables used above
eval_set = [(X_train, Y_train), (X_val, Y_val)]

# Train the model
XGBEngine.fit(
    X_train, Y_train,
    eval_set = eval_set,
    verbose  = True
)

# Retrieve training history
results = XGBEngine.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

suffix = f'{n_estimators}_{max_depth}_{learning_rate}_{early_stopping_rounds}'

hep.style.use("CMS")
# Plot training/validation logloss
fig, ax = plt.subplots(figsize=(14, 14))
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Validation')
ax.legend()
ax.set_ylabel(ylabel = 'Log Loss', fontsize = "medium")
ax.set_xlabel(xlabel = 'Number of Trees', fontsize = "medium")
ax.set_yscale('log')
ax.set_title(label = 'XGBoost Log Loss', fontsize = "large")
plt.savefig(f'{outdir}/logloss_{suffix}.png')
# plt.savefig(f'{outdir}/logloss_{suffix}.pdf')

####################################################################################
###### Training Accuracy ######
####################################################################################
from sklearn import metrics
train_accuracy = metrics.accuracy_score(Y_train, XGBEngine.predict(X_train))
val_accuracy   = metrics.accuracy_score(Y_val,   XGBEngine.predict(X_val))
print(f"Training Accuracy = {train_accuracy:.4f}")
print(f"Validation Accuracy = {val_accuracy:.4f}")

####################################################################################
###### ROC curve ######
####################################################################################

# Predict probabilities on train/val
y_train_pred_proba = XGBEngine.predict_proba(X_train)[:, 1]
y_val_pred_proba   = XGBEngine.predict_proba(X_val)[:, 1]

# ROC for train
fpr_train, tpr_train, _ = metrics.roc_curve(Y_train, y_train_pred_proba, pos_label=1)
roc_auc_train = metrics.auc(fpr_train, tpr_train)
rfpr_train = [1 - i for i in fpr_train]

# ROC for val
fpr_val, tpr_val, _ = metrics.roc_curve(Y_val, y_val_pred_proba, pos_label=1)
roc_auc_val = metrics.auc(fpr_val, tpr_val)
rfpr_val = [1 - i for i in fpr_val]

# np.savez(f'{outdir}/xgb_valroc.npz', tpr=tpr_val, fpr=fpr_val, auc=np.array([roc_auc_val], dtype=np.float32))

# Plot ROC
fig, ax = plt.subplots()
ax.set_title(label = 'Receiver Operating Characteristic', fontsize = "large")
ax.plot(tpr_train, np.array(rfpr_train), 'b', label=f'AUC (Train) = {roc_auc_train:.2f}')
ax.plot(tpr_val,   np.array(rfpr_val),   'g', label=f'AUC (Val)   = {roc_auc_val:.2f}')
ax.legend(loc='lower left')
ax.plot([0, 1], [1, 0], '--', color='gray')
ax.set_ylabel(ylabel = 'Signal Efficiency', fontsize = "medium")
ax.set_xlabel(xlabel = 'Background Rejection', fontsize = "medium")
plt.savefig(f'{outdir}/roc_{suffix}.png')
# plt.savefig(f'{outdir}/roc_{suffix}.pdf')

####################################################################################
###### Histogram comparison (validation set only) ######
####################################################################################
# Plot predicted scores for val signal vs background (reconstructed from Y_val)
sig_mask = Y_val == 1
bkg_mask = Y_val == 0
fig, ax = plt.subplots()
ax.hist(y_val_pred_proba[sig_mask], bins=60, label='Signal (Val)', histtype='step')
ax.hist(y_val_pred_proba[bkg_mask], bins=60, label='Background (Val)', histtype='step')
ax.set_yscale('log')
ax.set_xlabel(xlabel = "BDT response", fontsize = "medium")
ax.legend(loc='upper right')
plt.savefig(f'{outdir}/hist_{suffix}.png')
# plt.savefig(f'{outdir}/hist_{suffix}.pdf')

####################################################################################
###### Feature importance ######
####################################################################################
fig, ax = plt.subplots(figsize=(12, 12))
xgboost.plot_importance(XGBEngine, ax=ax)
plt.tight_layout()
plt.savefig(f'{outdir}/importance_{suffix}.png')
# plt.savefig(f'{outdir}/importance_{suffix}.pdf')

XGBEngine.save_model(f"{outdir}/WH_leptonic_classifier_{suffix}.json")

# ####################################################################################
# ###### Loss Function ######
# ####################################################################################
# results = XGBEngine.evals_result()
# epochs = len(results['validation_0']['logloss'])
# x_axis = range(0, epochs)

# fig, ax = plt.subplots()
# ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
# ax.plot(x_axis, results['validation_1']['logloss'], label='Validation')
# ax.legend()
# ax.set_ylabel('Log Loss')
# ax.set_yscale('log')
# ax.set_title('XGBoost Log Loss')
# plt.savefig(f'{outdir}/logloss.png')
# plt.savefig(f'{outdir}/logloss.pdf')
# ####################################################################################
# ###### Training Accuracy ######
# ####################################################################################
# train_accuracy = metrics.accuracy_score(Y_train, XGBEngine.predict(X_train))
# val_accuracy   = metrics.accuracy_score(Y_val,   XGBEngine.predict(X_val))
# print(f"Training Accuracy = {train_accuracy:.4f}")
# print(f"Validation Accuracy = {val_accuracy:.4f}")

# ####################################################################################
# ###### ROC curve ######
# ####################################################################################

# # Predict probabilities
# y_sig_train_pred = XGBEngine.predict_proba(X_sig_train)[:, 1]
# y_bkg_train_pred = XGBEngine.predict_proba(X_bkg_train)[:, 1]
# y_sig_val_pred   = XGBEngine.predict_proba(X_sig_val)[:, 1]
# y_bkg_val_pred   = XGBEngine.predict_proba(X_bkg_val)[:, 1]

# # ROC for train
# fpr_train, tpr_train, _ = metrics.roc_curve(
#     y_train, np.concatenate((y_sig_train_pred, y_bkg_train_pred)), pos_label=1)
# roc_auc_train = metrics.auc(fpr_train, tpr_train)
# rfpr_train = [1 - i for i in fpr_train]

# # ROC for validation
# fpr_val, tpr_val, _ = metrics.roc_curve(
#     y_val, np.concatenate((y_sig_val_pred, y_bkg_val_pred)), pos_label=1)
# roc_auc_val = metrics.auc(fpr_val, tpr_val)
# rfpr_val = [1 - i for i in fpr_val]

# np.savez(f'{outdir}/xgb_valroc.npz', tpr=tpr_val, fpr=fpr_val, auc=np.array([roc_auc_val], dtype=np.float32))

# # Plot ROC
# fig, ax = plt.subplots()
# ax.set_title('Receiver Operating Characteristic')
# ax.plot(tpr_train, np.array(rfpr_train), 'b', label=f'AUC (Train) = {roc_auc_train:.2f}')
# ax.plot(tpr_val,   np.array(rfpr_val),   'g', label=f'AUC (Val)   = {roc_auc_val:.2f}')
# ax.legend(loc='lower right')
# ax.plot([0, 1], [1, 0], '--', color='gray')
# ax.set_ylabel('Signal Efficiency')
# ax.set_xlabel('Background Rejection')
# plt.savefig(f'{outdir}/roc.png')
# plt.savefig(f'{outdir}/roc.pdf')

# ####################################################################################
# ###### Histogram comparison (validation set only) ######
# ####################################################################################
# fig, ax = plt.subplots()
# ax.hist(y_sig_val_pred, bins=60, label='Signal (Val)', histtype='step')
# ax.hist(y_bkg_val_pred, bins=60, label='Background (Val)', histtype='step')
# ax.set_yscale('log')
# ax.legend(loc='upper right')
# plt.savefig(f'{outdir}/hist.png')
# plt.savefig(f'{outdir}/hist.pdf')

# ####################################################################################
# ###### Feature importance ######
# ####################################################################################
# fig, ax = plt.subplots(figsize=(10, 8))
# xgboost.plot_importance(XGBEngine, ax=ax)
# plt.tight_layout()
# plt.savefig(f'{outdir}/importance.png')
# plt.savefig(f'{outdir}/importance.pdf')