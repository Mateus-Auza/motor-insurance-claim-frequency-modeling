2# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 19:29:53 2026

@author: Mateus Auza Cruz
"""

## Exploratory analysis

# Libraries


import pandas as pd
import numpy as np


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

import random

SEED = 29

np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

import os
#os.chdir("C:\\Users\\mtpla\\Downloads")


data = pd.read_csv("data/train_contrats.csv")

# -----------------------
# Basic summaries
# -----------------------

# Variable types
types = data.dtypes.to_frame(name="Type")

# Missing values
missing = data.isna().sum().to_frame(name="Missing")

# Numeric summary
desc = data.describe().T[["mean", "std", "min", "max"]]

# -----------------------
# Combine all
# -----------------------

summary = (
    types
    .join(missing)
    .join(desc)
)

# Round numeric columns safely
numeric_cols = [col for col in ["mean", "std", "min", "max"] if col in summary.columns]
summary[numeric_cols] = summary[numeric_cols].round(2)

# Clean output
summary = summary.reset_index().rename(columns={"index": "Variable"})

# Output
summary
##


## Feature enginering
y = data["nombre_de_sinistre"]
exposure = data["Exposition_au_risque"]

# Date variables converted into datetime format
data["Date_Deb_Situ"] = pd.to_datetime(data["Date_Deb_Situ"], dayfirst=True)
data["Date_Fin_Situ"] = pd.to_datetime(data["Date_Fin_Situ"], dayfirst= True)

data["year_start"] = data["Date_Deb_Situ"].dt.year
data["month_start"] = data["Date_Deb_Situ"].dt.month

data["duration_days"] = (
    data["Date_Fin_Situ"] - data["Date_Deb_Situ"]
).dt.days

# Numerical and logarithmic transformations
data["Valeur_assuree"] = pd.to_numeric(
    data["Valeur_assuree"], errors="coerce"
)

data["log_valeur_assuree"] = np.log1p(data["Valeur_assuree"])

data["vehicule_age_num"] = pd.to_numeric(
    data["Age_du_vehicule"], errors="coerce"
)

# Recmoval of variables
X = data.drop(columns=[
    "nombre_de_sinistre",    # What we want to predict
    "Num_contrat",           # Identifier 
    "IMMAT",                 # Identifier
    "Unnamed: 0",            # Identifier
    "Date_Deb_Situ",         # Because we already have the dateformat
    "Date_Fin_Situ",         # Because we already have the dateformat
    "Exposition_au_risque",  # used separately as offset
    "ANNEE",                 # Redundant variable because of the dateformat conversion
    "Categorie_ensemble",    # No variability -> Does not improve the model
    "Nombre_vehicule",       # No variability -> Does not improve the model
    "Valeur_assuree",        # We already have log_valeur_assure (numerical transformation)
    "Age_du_vehicule"        # We already have Age_du_vehicule_num (numerical transofrmation) 
])


X = pd.get_dummies(X, drop_first=True)

X = X.fillna(0) # Missing values are replaced with zeros.


X_train, X_val, y_train, y_val, exp_train, exp_val = train_test_split(
    X, y, exposure,
    test_size=0.2,
    random_state=42
)

X_val = X_val.reindex(columns=X_train.columns, fill_value=0)


#---------------------------------------------------------
#=========================================================
### Supervised learning ----------------------------------
#=========================================================
#---------------------------------------------------------


# GLM BENCHMARK

X_train_glm = sm.add_constant(X_train, has_constant='add')
X_val_glm = sm.add_constant(X_val, has_constant='add')


# Fix exposure
exp_train = exp_train.replace(0, 1e-8)
exp_val = exp_val.replace(0, 1e-8)


# Fit model
X_train_glm = X_train_glm.astype(float)
X_val_glm = X_val_glm.astype(float)

glm_model = sm.GLM(
    y_train,
    X_train_glm,
    family=sm.families.Poisson(),
    offset=np.log(exp_train) #,
    #freq_weights=exp_train   
)
glm_results = glm_model.fit()

print(glm_results.summary())

# Predictions (with offset)
y_pred_train = glm_results.predict(
    X_train_glm,
    offset=np.log(exp_train)
)

y_pred_val = glm_results.predict(
    X_val_glm,
    offset=np.log(exp_val)
)

# Poisson Deviance Function
def poisson_deviance(y, mu):
    y = np.array(y)
    mu = np.array(mu)

    # Avoid division/log issues
    mu = np.clip(mu, 1e-10, None)

    mask = y > 0
    term = np.zeros_like(y, dtype=float)

    term[mask] = y[mask] * np.log(y[mask] / mu[mask]) - (y[mask] - mu[mask])
    term[~mask] = -mu[~mask]

    return 2 * np.sum(term)

train_dev = poisson_deviance(y_train, y_pred_train)
val_dev = poisson_deviance(y_val, y_pred_val)

# Normalized deviance (better for comparison)
train_dev_norm = train_dev / len(y_train)
val_dev_norm = val_dev / len(y_val)

print("GLM Train Deviance:", train_dev)
print("GLM Validation Deviance:", val_dev)

print("GLM Train Deviance (normalized):", train_dev_norm)
print("GLM Validation Deviance (normalized):", val_dev_norm)

## NEURAL NETWORKS


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_nn = scaler.fit_transform(X_train)
X_val_nn = scaler.transform(X_val)



def build_model_1(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='exponential') 
    ])
    
    optimizer = keras.optimizers.Adam(
    learning_rate=0.0005,
    clipnorm=1.0
    )

    model.compile(optimizer=optimizer, loss='poisson')
    return model


def build_model_2(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='exponential')
    ])
    optimizer = keras.optimizers.Adam(
    learning_rate=0.0005,
    clipnorm=1.0
    )
    
    model.compile(optimizer=optimizer, loss='poisson')
    return model


def build_model_3(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='exponential')
    ])
    optimizer = keras.optimizers.Adam(
    learning_rate=0.0005,
    clipnorm=1.0
    )

    model.compile(optimizer=optimizer, loss='poisson')
    return model


def build_model_4(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='exponential')
    ])
    optimizer = keras.optimizers.Adam(
    learning_rate=0.0005,
    clipnorm=1.0
    )

    model.compile(optimizer=optimizer, loss='poisson')
    
    return model

models = [
    build_model_1(X_train_nn.shape[1]),
    build_model_2(X_train_nn.shape[1]),
    build_model_3(X_train_nn.shape[1]),
    build_model_4(X_train_nn.shape[1])
]

histories = []

for i, model in enumerate(models):
    history = model.fit(
    X_train_nn, y_train,                     
    sample_weight=exp_train,
    validation_data=(X_val_nn, y_val, exp_val),  
    epochs=20,
    batch_size=256,
    verbose=0
)
    

    histories.append(history)


results = []

for i, model in enumerate(models):
    y_pred_train = model.predict(X_train_nn, verbose=0).flatten()
    y_pred_val = model.predict(X_val_nn, verbose=0).flatten()

    train_dev = poisson_deviance(y_train, y_pred_train)
    val_dev = poisson_deviance(y_val, y_pred_val)
    norm_val_dev = val_dev / len(y_val)

    results.append({
        "Model": f"Model {i+1}",
        "Train Deviance": train_dev,
        "Validation Deviance": val_dev
    })

results_df = pd.DataFrame(results)
print(results_df)

# =========================
# Subgroup analysis
# =========================

# Validation dataset
val_df = data.loc[X_val.index].copy()

val_df["y_true"] = y_val.values
val_df["exposure"] = exp_val.values

# Align GLM input
X_val_glm = X_val_glm.loc[exp_val.index]
offset_val = np.log(exp_val.values)

# Predictions
val_df["glm_pred"] = glm_results.predict(X_val_glm, offset=offset_val)

best_model = models[2]
val_df["nn_pred"] = best_model.predict(X_val_nn, verbose=0).flatten()

# Exposure groups
val_df["exp_group"] = pd.cut(
    val_df["exposure"],
    bins=[0, 0.25, 0.5, 0.75, 1.0],
    labels=["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1"]
)

def safe_freq(num, denom):
    return num / denom if denom > 0 else np.nan

def compare_by_group(df, group_var, pred_cols):
    grouped = df.groupby(group_var, observed=True).apply(
        lambda g: pd.Series({
            "Observed freq": safe_freq(g["y_true"].sum(), g["exposure"].sum()),
            **{
                col: safe_freq(g[col].sum(), g["exposure"].sum())
                for col in pred_cols
            },
            "Count": len(g)
        }),
        include_groups=False
    ).reset_index()
    return grouped

# Global comparison
obs = val_df["y_true"].sum() / val_df["exposure"].sum()
glm_raw = val_df["glm_pred"].sum() / val_df["exposure"].sum()
nn_raw = val_df["nn_pred"].sum() / val_df["exposure"].sum()

bias_glm = obs / glm_raw
bias_nn = obs / nn_raw

val_df["glm_adj"] = val_df["glm_pred"] * bias_glm
val_df["nn_adj"] = val_df["nn_pred"] * bias_nn

glm_adj = val_df["glm_adj"].sum() / val_df["exposure"].sum()
nn_adj = val_df["nn_adj"].sum() / val_df["exposure"].sum()

global_table = pd.DataFrame({
    "Observed": [obs],
    "GLM (raw)": [glm_raw],
    "NN (raw)": [nn_raw],
    "GLM (adjusted)": [glm_adj],
    "NN (adjusted)": [nn_adj]
})

print(global_table)

# Group comparisons
exp_comp = compare_by_group(
    val_df, "exp_group",
    ["glm_pred", "nn_pred", "glm_adj", "nn_adj"]
)

print(exp_comp)
area_comp = compare_by_group(
    val_df, "Zone",
    ["glm_pred", "nn_pred", "glm_adj", "nn_adj"]
)
print(area_comp)

activity_comp = compare_by_group(
    val_df, "Activite",
    ["glm_pred", "nn_pred", "glm_adj", "nn_adj"]
)
print(activity_comp)

# =========================
# PDP & ICE
# =========================

from sklearn.inspection import PartialDependenceDisplay, partial_dependence, permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin

class NNWrapper(RegressorMixin, BaseEstimator):
    _estimator_type = "regressor"

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        return self

    def predict(self, X):
        X = X.astype(np.float64)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, batch_size=1024, verbose=0).flatten()

nn_model = NNWrapper(best_model, scaler)
_ = nn_model.fit(X_train)

X_train_pd = X_train.sample(n=2000, random_state=0).astype(np.float64)

# Feature importance
X_sample = X_train.sample(n=1000, random_state=0).astype(np.float64)

r = permutation_importance(
    nn_model,
    X_sample,
    nn_model.predict(X_sample),
    n_repeats=2,
    random_state=0,
    n_jobs=1
)

importance_nn = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": r.importances_mean
}).sort_values(by="Importance", ascending=False)

top_features = importance_nn.head(5)["Feature"].tolist()

# PDP & ICE plots
subset = top_features[:3]
n = len(subset)

fig, axes = plt.subplots(n, 2, figsize=(8, 2*n))

for i, feature in enumerate(subset):
    PartialDependenceDisplay.from_estimator(
        nn_model, X_train_pd, [feature], kind="average", ax=axes[i, 0]
    )
    axes[i, 0].set_title(f"PDP - {feature}")

    PartialDependenceDisplay.from_estimator(
        nn_model, X_train_pd, [feature], kind="individual",
        subsample=50, random_state=0, ax=axes[i, 1]
    )
    axes[i, 1].set_title(f"ICE - {feature}")

plt.tight_layout()
plt.show()

subset = top_features[3:]
n = len(subset)

fig, axes = plt.subplots(n, 2, figsize=(8, 2*n))

for i, feature in enumerate(subset):
    PartialDependenceDisplay.from_estimator(
        nn_model, X_train_pd, [feature], kind="average", ax=axes[i, 0]
    )
    axes[i, 0].set_title(f"PDP - {feature}")

    PartialDependenceDisplay.from_estimator(
        nn_model, X_train_pd, [feature], kind="individual",
        subsample=50, random_state=0, ax=axes[i, 1]
    )
    axes[i, 1].set_title(f"ICE - {feature}")

plt.tight_layout()
plt.show()

# PDP range importance
feature_importance = []

for feature in top_features:
    pd_result = partial_dependence(nn_model, X_train_pd, [feature])
    pd_values = pd_result["average"][0]
    importance = np.max(pd_values) - np.min(pd_values)
    feature_importance.append((feature, importance))

importance_df = pd.DataFrame(
    feature_importance,
    columns=["Feature", "PDP Range"]
).sort_values(by="PDP Range", ascending=False)


# =========================
# LIME & SHAP
# =========================

import shap
from lime.lime_tabular import LimeTabularExplainer

X_train_num = X_train.astype(np.float64)
X_val_num = X_val.astype(np.float64)

def predict_fn(X):
    X = pd.DataFrame(X, columns=X_train.columns)
    X = X.astype(np.float64)
    X_scaled = scaler.transform(X)
    return best_model.predict(X_scaled, verbose=0).flatten()

# Select two policies based on prediction based on the best NN model (model 3)
preds = predict_fn(X_val_num)

low_idx = preds.argmin()
high_idx = preds.argmax()

policy_1 = X_val_num.iloc[[low_idx]]
policy_2 = X_val_num.iloc[[high_idx]]

# LIME
explainer_lime = LimeTabularExplainer(
    training_data=X_train_num.values,
    feature_names=X_train.columns.tolist(),
    mode='regression',
    discretize_continuous=True
)

exp1 = explainer_lime.explain_instance(policy_1.values[0], predict_fn, num_features=10)
exp2 = explainer_lime.explain_instance(policy_2.values[0], predict_fn, num_features=10)

def lime_to_df(exp):
    df = pd.DataFrame(exp.as_list(), columns=["Condition", "Contribution"])
    df["Impact"] = df["Contribution"].apply(lambda x: "↑ Increase" if x > 0 else "↓ Decrease")
    return df.sort_values(by="Contribution", key=abs, ascending=False)

df1 = lime_to_df(exp1)
df2 = lime_to_df(exp2)

# SHAP
background = X_train_num.sample(100, random_state=0).values

explainer_shap = shap.Explainer(
    lambda x: predict_fn(pd.DataFrame(x, columns=X_train.columns)),
    background
)

policy_1_np = policy_1.values.astype(np.float64)
policy_2_np = policy_2.values.astype(np.float64)

shap_values_1 = explainer_shap(policy_1_np)
shap_values_2 = explainer_shap(policy_2_np)

shap.plots.waterfall(shap_values_1[0])
shap.plots.waterfall(shap_values_2[0])

# Comparison
pred1 = predict_fn(policy_1)[0]
pred2 = predict_fn(policy_2)[0]

comparison = pd.DataFrame({
    "Feature": X_train.columns,
    "Policy_1_SHAP": shap_values_1.values[0],
    "Policy_2_SHAP": shap_values_2.values[0]
})

comparison["Difference"] = comparison["Policy_1_SHAP"] - comparison["Policy_2_SHAP"]

comparison_sorted = comparison.iloc[
    comparison["Difference"].abs().sort_values(ascending=False).index
].head(10)



# =========================
# Unsupervised learning - Data preparation
# =========================

import keras.ops as ops
from sklearn.preprocessing import StandardScaler

features_unsup = [
    "Type_Apporteur",
    "Fractionnement",
    "Creation_Entr",
    "Age_du_vehicule",
    "Mode_gestion",
    #"ValeurPuissance",
    "Activite",
    "FORMULE",
    "Zone",
    "Segment"
]

data_unsup = data[features_unsup].copy()

#data_unsup["Age_du_vehicule"] = pd.to_numeric(
#    data_unsup["Age_du_vehicule"], errors="coerce"
#)

data_unsup = pd.get_dummies(data_unsup, drop_first=True)
data_unsup = data_unsup.fillna(0)

scaler_unsup = StandardScaler()
X_unsup = scaler_unsup.fit_transform(data_unsup)

input_dim = X_unsup.shape[1]


# =========================
# Autoencoders
# =========================

def build_autoencoder_1(input_dim):
    inputs = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(32, activation='relu')(inputs)
    encoded = layers.Dense(8, activation='relu')(encoded)
    decoded = layers.Dense(32, activation='relu')(encoded)
    outputs = layers.Dense(input_dim, activation='linear')(decoded)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_autoencoder_2(input_dim):
    inputs = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(64, activation='relu')(inputs)
    encoded = layers.Dropout(0.2)(encoded)
    encoded = layers.Dense(16, activation='relu')(encoded)
    decoded = layers.Dense(64, activation='relu')(encoded)
    outputs = layers.Dense(input_dim, activation='linear')(decoded)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


# =========================
# Variational Autoencoders
# =========================

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def compute_losses(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)

        reconstruction_loss = ops.mean(
            ops.square(data - reconstruction), axis=1
        )

        kl_loss = -0.5 * ops.sum(
            1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var),
            axis=1
        )

        return ops.mean(reconstruction_loss + kl_loss)

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            total_loss = self.compute_losses(data)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {"loss": total_loss}

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        return {"loss": self.compute_losses(data)}


def build_vae_1(input_dim, latent_dim=2):
    inputs = keras.Input(shape=(input_dim,))
    h = layers.Dense(32, activation='relu')(inputs)

    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(inputs, [z_mean, z_log_var, z])

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(32, activation='relu')(latent_inputs)
    outputs = layers.Dense(input_dim, activation='linear')(x)

    decoder = keras.Model(latent_inputs, outputs)

    vae = VAE(encoder, decoder)
    vae.compile(optimizer='adam')

    return vae


def build_vae_2(input_dim, latent_dim=4):
    inputs = keras.Input(shape=(input_dim,))
    h = layers.Dense(64, activation='relu')(inputs)
    h = layers.Dense(32, activation='relu')(h)

    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(inputs, [z_mean, z_log_var, z])

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(32, activation='relu')(latent_inputs)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(input_dim, activation='linear')(x)

    decoder = keras.Model(latent_inputs, outputs)

    vae = VAE(encoder, decoder)
    vae.compile(optimizer='adam')

    return vae


# =========================
# Training & evaluation
# =========================

models_unsup = {
    "AE1": build_autoencoder_1(input_dim),
    "AE2": build_autoencoder_2(input_dim),
    "VAE1": build_vae_1(input_dim),
    "VAE2": build_vae_2(input_dim)
}

histories_unsup = {}

for name, model in models_unsup.items():
    history = model.fit(
        X_unsup, X_unsup,
        epochs=30,
        batch_size=256,
        validation_split=0.2,
        verbose=0
    )
    histories_unsup[name] = history

def reconstruction_error(model, X):
    X_pred = model.predict(X, verbose=0)
    return np.mean(np.square(X - X_pred))

results_unsup = []

for name, model in models_unsup.items():
    error = reconstruction_error(model, X_unsup)
    results_unsup.append({
        "Model": name,
        "Reconstruction MSE": error
    })

results_unsup_df = pd.DataFrame(results_unsup)


# =========================
# K-means clustering
# =========================

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X = X_unsup

inertia = []
silhouette_scores = []
K_range = range(2, 21)

sample_size = min(5000, len(X))
np.random.seed(42)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    inertia.append(kmeans.inertia_)

    idx = np.random.choice(len(X), sample_size, replace=False)
    score = silhouette_score(X[idx], labels[idx])
    silhouette_scores.append(score)

best_k = K_range[np.argmax(silhouette_scores)]

plt.figure()
plt.plot(K_range, inertia, marker='o')
plt.axvline(best_k, linestyle='--')
plt.show()

plt.figure()
plt.plot(K_range, silhouette_scores, marker='o')
plt.scatter(best_k, max(silhouette_scores))
plt.axvline(best_k, linestyle='--')
plt.show()


# Final clustering
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X)

clusters_series = pd.Series(clusters, index=data.index)

data_clusters = data_unsup.copy()
data_clusters["Cluster"] = clusters_series

cluster_sizes = data_clusters["Cluster"].value_counts().sort_index()
cluster_sizes_df = cluster_sizes.to_frame().T
cluster_sizes_df.columns = [f"C{c}" for c in cluster_sizes_df.columns]

cluster_summary = data_clusters.groupby("Cluster").mean().round(2)

one_hot_cols = data_unsup.columns

rows = []
for cluster in sorted(data_clusters["Cluster"].unique()):
    means = data_clusters[data_clusters["Cluster"] == cluster][one_hot_cols].mean()
    top_features = means.sort_values(ascending=False).head(5)
    rows.append([cluster] + list(top_features.index))

dominant_compact = pd.DataFrame(
    rows,
    columns=["Cluster", "Top 1", "Top 2", "Top 3", "Top 4", "Top 5"]
)


# =========================
# GLM vs NN vs Clusters
# =========================

df_cluster_full = pd.DataFrame({
    "y": y.values,
    "exposure": exposure.values,
    "cluster": clusters_series.values
})

cluster_stats = df_cluster_full.groupby("cluster")[["y", "exposure"]].sum()

cluster_stats["frequency"] = (
    cluster_stats["y"] / cluster_stats["exposure"]
)

alpha = 100

global_freq = (
    df_cluster_full["y"].sum() /
    df_cluster_full["exposure"].sum()
)

cluster_stats["frequency_smooth"] = (
    cluster_stats["y"] + alpha * global_freq
) / (
    cluster_stats["exposure"] + alpha
)

df_cluster_val = pd.DataFrame({
    "y": y_val.values,
    "exposure": exp_val.values,
    "cluster": clusters_series.loc[X_val.index].values
})

df_cluster_val["freq_pred"] = df_cluster_val["cluster"].map(
    cluster_stats["frequency_smooth"]
)

df_cluster_val["mu_cluster"] = (
    df_cluster_val["freq_pred"] *
    df_cluster_val["exposure"]
)

cluster_pred_val = df_cluster_val["mu_cluster"].values

def poisson_deviance(y, mu):
    y = np.array(y)
    mu = np.maximum(np.array(mu), 1e-10)

    mask = y > 0
    dev = np.zeros_like(y, dtype=float)

    ratio = np.zeros_like(y, dtype=float)
    ratio[mask] = y[mask] / mu[mask]
    ratio[mask] = np.maximum(ratio[mask], 1e-10)

    dev[mask] = y[mask] * np.log(ratio[mask]) - (y[mask] - mu[mask])
    dev[~mask] = -mu[~mask]

    return 2 * np.sum(dev)

glm_pred_val = glm_results.predict(
    X_val_glm,
    offset=np.log(exp_val)
)

nn_pred_val = models[0].predict(X_val_nn, verbose=0).flatten()

baseline_mu_val = exp_val * global_freq

dev_glm = poisson_deviance(y_val, glm_pred_val)
dev_nn = poisson_deviance(y_val, nn_pred_val)
dev_cluster = poisson_deviance(y_val, cluster_pred_val)
dev_baseline = poisson_deviance(y_val, baseline_mu_val)

n = len(y_val)

results_dev = pd.DataFrame({
    "Model": ["GLM", "Neural Network", "Clusters", "Baseline"],
    "Deviance": [dev_glm, dev_nn, dev_cluster, dev_baseline],
    "Mean Deviance": [dev_glm/n, dev_nn/n, dev_cluster/n, dev_baseline/n]
}).sort_values("Deviance")


print(results_dev.sort_values("Deviance"))






