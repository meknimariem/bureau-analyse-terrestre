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