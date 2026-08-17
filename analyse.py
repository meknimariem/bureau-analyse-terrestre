import os
import time
import urllib.request
import pandas as pd
from partie3 import executer_partie3

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

# ===================== PHASE 6 : le modèle bête du stagiaire =====================
import numpy as np
from sklearn.metrics import accuracy_score

# Le stagiaire répond "pas un canular" à tout le monde
pred_stagiaire = np.zeros(len(yh_test), dtype=bool)

acc_stagiaire = accuracy_score(yh_test, pred_stagiaire)
acc_modele = accuracy_score(yh_test, pred_h)

print("\n===== PHASE 6 : stagiaire vs vrai modèle =====")
print(f"Stagiaire : exactitude {100*acc_stagiaire:.1f} %  |  rappel "
      f"{100*recall_score(yh_test, pred_stagiaire):.1f} %")
print(f"Ton modèle: exactitude {100*acc_modele:.1f} %  |  rappel "
      f"{100*rappel_apres:.1f} %")


# ===================== PHASE 7 : plusieurs témoins, un seul événement =====================
from sklearn.model_selection import GroupShuffleSplit

# Helper réutilisé aux phases suivantes : un modèle honnête neuf à chaque appel
def construire_modele():
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])

# Clé d'événement : même ville + même date-heure d'observation
df["event"] = df["city"].fillna("?") + " | " + df["datetime"].fillna("?")
tailles = df["event"].value_counts()

print("\n===== PHASE 7 : témoins multiples =====")
print(f"Événements signalés par +1 témoin : {int((tailles > 1).sum())}")
print(f"Témoins pour le plus gros          : {int(tailles.max())}  ({tailles.idxmax()})")

com_nv = df["comments"].fillna("")
print(f"Témoignages en doublon exact       : {int((com_nv.duplicated(keep=False) & (com_nv != '')).sum())}")

# Relevés à cheval dans la découpe au hasard (Phase 5)
ev_app = set(df.loc[Xh_app.index, "event"])
ev_test = set(df.loc[Xh_test.index, "event"])
a_cheval = ev_app & ev_test
print(f"Relevés à cheval (découpe hasard)  : {int(df['event'].isin(a_cheval).sum())}")

# Nouvelle découpe : chaque événement entièrement d'un seul côté
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
i_app, i_test = next(gss.split(X_h, y_h, groups=df["event"]))
modele_g = construire_modele().fit(X_h.iloc[i_app], y_h.iloc[i_app])
pred_g = modele_g.predict(X_h.iloc[i_test])
rappel_groupe = recall_score(y_h.iloc[i_test], pred_g)
precision_groupe = precision_score(y_h.iloc[i_test], pred_g)

print("\nDeux nombres de la phase 4 :")
print(f"  découpe au hasard : rappel {100*rappel_apres:.1f} % | précision {100*precision_apres:.2f} %")
print(f"  découpe par groupe: rappel {100*rappel_groupe:.1f} % | précision {100*precision_groupe:.2f} %")

# Preuve : un événement entier, tous ses témoins du même côté
gros = tailles.idxmax()
pos = np.where((df["event"] == gros).values)[0]
cote = ("apprentissage" if set(pos).issubset(set(i_app))
        else "test" if set(pos).issubset(set(i_test)) else "À CHEVAL")
print(f"\nLes {len(pos)} témoins de '{gros}' sont tous côté : {cote}")
print(df.loc[df["event"] == gros, ["city", "datetime", "shape", "canular"]].to_string())



# =====================================================================
# ============  PARTIE 2 : corrections du Conseil (8 à 12)  ============
# =====================================================================
import re
from sklearn.preprocessing import StandardScaler

# ---------- PRÉPARATION DES FEATURES (silencieux : rien n'est "appris" ici) ----------
for c in ["latitude", "longitude", "duration_seconds"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
obs = pd.to_datetime(df["datetime"].str.replace(" 24:00", " 00:00", regex=False),
                     errors="coerce", format="mixed")
df["obs_date"] = obs
_h = obs.dt.hour
df["heure_sin"] = np.sin(2 * np.pi * _h / 24)     # (Phase 12) heure cyclique
df["heure_cos"] = np.cos(2 * np.pi * _h / 24)

def texte_en_secondes(t):                          # (Phase 11) parse du texte témoin
    if not isinstance(t, str): return np.nan
    s = t.lower(); m = re.search(r'(\d+(?:[.,]\d+)?)', s.replace('/', '.'))
    if not m: return np.nan
    v = float(m.group(1).replace(',', '.'))
    return v * (1 if re.search(r'sec', s) else 60 if re.search(r'min', s)
                else 3600 if re.search(r'hr|hour', s) else 86400 if re.search(r'day', s) else 60)

sec_txt = df["duration_hours_min"].apply(texte_en_secondes)
df["duree"] = df["duration_seconds"].copy()
recuperes = ((df["duree"].isna()) | (df["duree"] == 0)) & sec_txt.notna()
df.loc[recuperes, "duree"] = sec_txt[recuperes]
df["duree"] = df["duree"].clip(upper=86400)        # décision : plafond 1 journée

df["shape2"] = df["shape"].replace({"changed": "changing", "round": "circle"})  # (Phase 12)
_cpt = df["shape2"].value_counts()
df["shape2"] = df["shape2"].where(~df["shape2"].isin(_cpt[_cpt < 10].index), "other")

# ==================== PHASE 8 : découpe temporelle ====================
date_coupure = df["obs_date"].quantile(0.8)
app = df["obs_date"] <= date_coupure
tst = df["obs_date"] > date_coupure
print("\n===== PHASE 8 : découpe temporelle =====")
print(f"Date de coupure       : {date_coupure}")
print(f"Relevés app / test    : {int(app.sum())} / {int(tst.sum())}")
print(f"Prop canulars app     : {100*df.loc[app,'canular'].mean():.3f} %")
print(f"Prop canulars test    : {100*df.loc[tst,'canular'].mean():.3f} %")
# ville réduite : fréquences apprises sur l'APPRENTISSAGE seul (pas de fuite)
vc_app = df.loc[app, "city"].value_counts()
df["ville"] = df["city"].where(df["city"].map(vc_app).fillna(0) >= 20, "autre")

# ==================== PHASE 9 : les trous comme signal ====================
print("\n===== PHASE 9 : trous = signal =====")
manquant = df[COLS].replace("", np.nan).isna().sum().sort_values(ascending=False)
top3 = manquant.head(3).index.tolist()
print("Trois colonnes les plus trouées :", top3)
for c in top3:
    trou = df[c].replace("", np.nan).isna()
    print(f"  {c:20} AVEC trou {100*df.loc[trou,'canular'].mean():.3f}% | "
          f"SANS trou {100*df.loc[~trou,'canular'].mean():.3f}%")
for c in ["country", "state", "duration_hours_min"]:      # traitement : indicateur de trou
    df[c + "_manque"] = df[c].replace("", np.nan).isna().astype(int)

    # ==================== PHASE 10 : pipeline unique, sans fuite ====================
num = ["duree", "latitude", "longitude", "heure_sin", "heure_cos",
       "country_manque", "state_manque", "duration_hours_min_manque"]
cat = ["shape2", "country", "ville"]
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
])
modele_final = Pipeline([("pre", pre),
                         ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])
modele_final.fit(df.loc[app, num + cat], df.loc[app, "canular"])   # fit sur APP seul
pred_final = modele_final.predict(df.loc[tst, num + cat])
largeur = modele_final.named_steps["pre"].transform(df.loc[app, num + cat]).shape[1]
print("\n===== PHASE 10 : pipeline final =====")
print(f"Largeur du tableau après encodage : {largeur} colonnes")
print(f"Rappel FINAL    : {100*recall_score(df.loc[tst,'canular'], pred_final):.1f} %")
print(f"Précision FINAL : {100*precision_score(df.loc[tst,'canular'], pred_final):.2f} %")
print("Prédiction sur 1 relevé neuf :", modele_final.predict(df.loc[tst, num + cat].iloc[[0]])[0])


# ==================== PHASE 11 : réconcilier les durées ====================
both = df["duration_seconds"].notna() & (df["duration_seconds"] > 0) & sec_txt.notna()
contra = both & ((df["duration_seconds"] - sec_txt).abs()
                 / df["duration_seconds"].where(df["duration_seconds"] > 0) > 0.5)
print("\n===== PHASE 11 : durées =====")
print(f"Encore inutilisable          : {int(df['duree'].isna().sum())}")
print(f"Récupérés (secondes=0 + texte): {int(recuperes.sum())}")
print(f"Contradictions               : {int(contra.sum())}")
print(f"Durée médiane                : {df['duree'].median():.0f} s")
print(f"Relevés > 1 journée          : {int((df['duree'] >= 86400).sum())}")
# ==================== PHASE 12 : ville, heure, forme ====================
def dist(a, b):
    ea = np.array([np.sin(2*np.pi*a/24), np.cos(2*np.pi*a/24)])
    eb = np.array([np.sin(2*np.pi*b/24), np.cos(2*np.pi*b/24)])
    return np.linalg.norm(ea - eb)
vc_tot = df["city"].value_counts()
print("\n===== PHASE 12 : ville, heure, forme =====")
print(f"Villes : {df['city'].nunique()} distinctes -> {df['ville'].nunique()} colonnes "
      f"({int((vc_tot==1).sum())} vues 1 seule fois)")
print(f"Distance 23h<->0h : {dist(23,0):.3f} | Distance 23h<->20h : {dist(23,20):.3f}")
print(f"Formes : 29 -> {df['shape2'].nunique()}")
# ===================== PARTIE 3 (module séparé) =====================
executer_partie3(df, num, cat, app, tst, modele_final)