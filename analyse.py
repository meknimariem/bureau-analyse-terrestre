import os
import time
import urllib.request
import pandas as pd

CSV = "releves_klaxo3.csv"
URL = ("https://cdn.jsdelivr.net/gh/planetsig/ufo-reports@master/"
       "csv-data/ufo-complete-geocoded-time-standardized.csv")

def telecharger(url, dest, essais=4):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for i in range(essais):
        try:
            with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
                f.write(r.read())
            return
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < essais - 1:
                attente = 5 * (i + 1)
                print(f"Serveur saturé (429), nouvelle tentative dans {attente}s...")
                time.sleep(attente)
            else:
                raise

if not os.path.exists(CSV):
    print("Téléchargement des relevés...")
    telecharger(URL, CSV)

COLS = ["datetime", "city", "state", "country", "shape",
        "duration_seconds", "duration_hours_min", "comments",
        "date_posted", "latitude", "longitude"]

# ===================== PHASE 1 : ouvrir la caisse =====================
# 1) Nombre de lignes dans le fichier brut (compte physique, robuste à l'encodage)
with open(CSV, "rb") as f:
    total = sum(1 for _ in f)

# 2) Chargement en gardant la trace des lignes que pandas refuse
lignes_ecartees = []
def garder_trace(bad_line):
    lignes_ecartees.append(bad_line)
    return None          # None => la ligne est ignorée (mais on l'a notée)

df = pd.read_csv(CSV, header=None, names=COLS,
                 engine="python", on_bad_lines=garder_trace)

chargees = len(df)
ecartees = len(lignes_ecartees)

print(f"Lignes dans le fichier : {total}")
print(f"Lignes chargées        : {chargees}")
print(f"Lignes mises de côté   : {ecartees}")
print(f"Contrôle : {chargees} + {ecartees} = {chargees + ecartees}")

# 3) Afficher une ligne problématique pour comprendre ce qui cloche
print("\nExemple de ligne écartée :")
print(lignes_ecartees[0])
# ===================== PHASE 2 : chaque champ dans son vrai type =====================
# On ne supprime AUCUNE ligne ici. On convertit et on compte ce qui résiste.

def compter_echecs(serie, serie_convertie):
    """Valeurs non nulles au départ qui deviennent nulles après conversion."""
    fautives = serie[serie_convertie.isna() & serie.notna()]
    return fautives

print("\n===== PHASE 2 : conversion des types =====")

# --- Démonstration : une seule valeur casse toute la colonne latitude ---
try:
    pd.to_numeric(df["latitude"])                 # sans filet
except ValueError as e:
    print(f"\nlatitude sans 'coerce' -> PLANTE : {e}")

# --- Colonnes numériques ---
for c in ["latitude", "longitude", "duration_seconds"]:
    conv = pd.to_numeric(df[c], errors="coerce")
    fautives = compter_echecs(df[c], conv)
    print(f"\n{c} : {len(fautives)} valeur(s) non convertible(s)")
    if len(fautives):
        print("   exemples :", fautives.unique()[:5].tolist())

# --- Colonnes dates ---
# datetime : on répare d'abord le '24:00' (minuit) qui n'existe pas
datetime_repare = df["datetime"].str.replace(" 24:00", " 00:00", regex=False)
conv_dt = pd.to_datetime(datetime_repare, errors="coerce", format="mixed")
h24 = df["datetime"].str.contains(" 24:00", na=False).sum()
print(f"\ndatetime : {h24} valeur(s) en '24:00' (réparées en '00:00')")

conv_dp = pd.to_datetime(df["date_posted"], errors="coerce", format="mixed")
fautives_dp = compter_echecs(df["date_posted"], conv_dp)
print(f"date_posted : {len(fautives_dp)} valeur(s) non convertible(s)")

# --- Anomalie sémantique : coordonnées (0,0) = Null Island ---
lat = pd.to_numeric(df["latitude"], errors="coerce")
lon = pd.to_numeric(df["longitude"], errors="coerce")
null_island = ((lat == 0) & (lon == 0)).sum()
print(f"\nCoordonnées (0,0) 'Null Island' : {null_island}")

# --- Contrôle : aucune ligne perdue ---
print(f"\nNombre de lignes inchangé : {len(df)}")
# ===================== PHASE 3 : fabriquer l'étiquette "canular" =====================
# Règle : un relevé est un canular si son témoignage contient le mot "hoax".
com = df["comments"].fillna("").str.lower()
df["canular"] = com.str.contains("hoax", regex=False)

print("\n===== PHASE 3 : étiquette canular =====")
print("Règle : le témoignage (comments) contient 'hoax'")
print(f"Canulars marqués : {df['canular'].sum()}")
print(f"Proportion       : {100 * df['canular'].mean():.3f} %")

# Limite 1 — attrape à tort : "not a hoax" est marqué canular
faux_pos = com.str.contains("not a hoax", regex=False)
print(f"\nAttrapés à tort ('not a hoax') : {faux_pos.sum()}")
print("   ex :", df["comments"][faux_pos].iloc[0][:80])

# Limite 2 — rate : canulars annotés 'fake' mais pas 'hoax'
rate = com.str.contains("fake", regex=False) & ~df["canular"]
print(f"Ratés (marqués 'fake' sans 'hoax') : {rate.sum()}")
# ===================== PHASE 4 : premier verdict (rappel & précision) =====================
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score

X = df["comments"].fillna("")
y = df["canular"]

# Découpe apprentissage / test (20 % pour tester, stratifiée pour garder des canulars des deux côtés)
X_app, X_test, y_app, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Le modèle apprend le texte des témoignages
vec = TfidfVectorizer(max_features=5000)
X_app_v = vec.fit_transform(X_app)
X_test_v = vec.transform(X_test)

modele = LogisticRegression(max_iter=1000, class_weight="balanced")
modele.fit(X_app_v, y_app)
pred = modele.predict(X_test_v)

print("\n===== PHASE 4 : premier verdict =====")
print(f"Relevés de test (jamais vus) : {len(y_test)} dont {y_test.sum()} canulars")
print(f"Rappel    : {100 * recall_score(y_test, pred):.1f} %")
print(f"Précision : {100 * precision_score(y_test, pred):.1f} %")
# ===================== PHASE 5 : démasquer la fuite de données =====================
# On mémorise d'abord les deux nombres de la Phase 4 (modèle AVEC comments)
rappel_avant = recall_score(y_test, pred)
precision_avant = precision_score(y_test, pred)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Colonnes "honnêtes" : écrites AVANT tout jugement de canular
df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce")
df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
dt = pd.to_datetime(df["datetime"].str.replace(" 24:00", " 00:00", regex=False),
                    errors="coerce", format="mixed")
df["heure"] = dt.dt.hour

num = ["duration_seconds", "latitude", "longitude", "heure"]
cat = ["shape", "country"]
X_h, y_h = df[num + cat], df["canular"]

Xh_app, Xh_test, yh_app, yh_test = train_test_split(
    X_h, y_h, test_size=0.2, random_state=42, stratify=y_h)

pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
])
modele_h = Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
modele_h.fit(Xh_app, yh_app)
pred_h = modele_h.predict(Xh_test)

rappel_apres = recall_score(yh_test, pred_h)
precision_apres = precision_score(yh_test, pred_h)

print("\n===== PHASE 5 : avant / après retrait de comments =====")
print(f"Rappel    : {100*rappel_avant:5.1f} % (avant)  ->  {100*rappel_apres:5.1f} % (après)")
print(f"Précision : {100*precision_avant:5.1f} % (avant)  ->  {100*precision_apres:5.1f} % (après)")