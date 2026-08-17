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