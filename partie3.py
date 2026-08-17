# partie3.py — Phases 13 à 18 : défendre une décision
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score, precision_score


def executer_partie3(df, num, cat, app, tst, modele_final):
    df["annee"] = df["obs_date"].dt.year
    Xte = df.loc[tst, num + cat]
    yte = df.loc[tst, "canular"].values
    proba = modele_final.predict_proba(Xte)[:, 1]

    def construire_pipeline_final():
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
        return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])

    # ===== PHASE 13 : seuil au moindre coût (FN=30, FP=2) =====
    def facture(seuil):
        pred = proba >= seuil
        fn = int(((yte == True) & (~pred)).sum())
        fp = int(((yte == False) & (pred)).sum())
        return 30 * fn + 2 * fp
    seuils = np.round(np.arange(0.05, 0.991, 0.05), 2)
    couts = [(s, facture(s)) for s in seuils]
    meilleur = min(couts, key=lambda x: x[1])
    print("\n===== PHASE 13 : facture =====")
    for s, c in couts:
        print(f"  seuil {s:.2f} -> {c} crédits")
    print(f"Facture à 0.5    : {facture(0.5)} crédits")
    print(f"Meilleur seuil   : {meilleur[0]} -> {meilleur[1]} crédits")
    print(f"Écart (économie) : {facture(0.5) - meilleur[1]} crédits")
    SEUIL = meilleur[0]

    # ===== PHASE 14 : calibration =====
    def table_calibration(p):
        b = pd.cut(p, bins=[0, .2, .4, .6, .8, 1.0])
        return (pd.DataFrame({"proba": p, "vrai": yte}).groupby(b, observed=True)
                .agg(n=("vrai", "size"), proba_moyenne=("proba", "mean"), reel=("vrai", "mean")).round(3))
    print("\n===== PHASE 14 : calibration =====")
    print("AVANT (brut) :"); print(table_calibration(proba).to_string())
    cal = CalibratedClassifierCV(construire_pipeline_final(), method="isotonic", cv=3)
    cal.fit(df.loc[app, num + cat], df.loc[app, "canular"])
    proba_cal = cal.predict_proba(Xte)[:, 1]
    print("APRÈS calibration isotonique :"); print(table_calibration(proba_cal).to_string())

    # ===== PHASE 15 : fourchette sur plusieurs découpes =====
    X_all, y_all = df[num + cat], df["canular"]
    rappels = []
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    for i_tr, i_te in sss.split(X_all, y_all):
        m = construire_pipeline_final().fit(X_all.iloc[i_tr], y_all.iloc[i_tr])
        rappels.append(recall_score(y_all.iloc[i_te], m.predict(X_all.iloc[i_te])))
    rappels = np.array(rappels)
    print("\n===== PHASE 15 : fourchette =====")
    print(f"Rappel : {100*rappels.mean():.1f} % (fourchette {100*rappels.min():.1f} – {100*rappels.max():.1f} %, 10 découpes)")
    print(f"Taille test : {len(yte)} | canulars dedans : {int(yte.sum())}")

    # ===== PHASE 16 : explicabilité =====
    pi = permutation_importance(modele_final, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=0)
    classement = pd.Series(pi.importances_mean, index=num + cat).sort_values(ascending=False)
    print("\n===== PHASE 16 : classement des colonnes =====")
    print(classement.round(4).to_string())
    sub = df.loc[tst].copy(); sub["proba"] = proba
    cas_sur = sub[sub["canular"]].sort_values("proba", ascending=False).iloc[0]
    cas_limite = sub[sub["proba"] >= SEUIL].sort_values("proba").iloc[0]
    cas_rate = sub[sub["canular"] & (sub["proba"] < SEUIL)].sort_values("proba").iloc[0]
    for nom, cas in [("SÛR", cas_sur), ("LIMITE", cas_limite), ("RATÉ", cas_rate)]:
        h = cas["obs_date"].hour if pd.notna(cas["obs_date"]) else "?"
        print(f"[{nom}] proba={cas['proba']:.3f} | ville={cas['city']} shape={cas['shape2']} heure={h} duree={cas['duree']}")

    # ===== PHASE 17 : par zone =====
    def zone(c):
        if c == "us": return "USA"
        if c in ("ca", "gb", "au"): return "Anglo (CA/GB/AU)"
        if pd.isna(c) or c == "": return "Inconnu"
        return "Autre"
    z = df.loc[tst].copy(); z["zone"] = z["country"].apply(zone); z["pred"] = proba >= SEUIL
    print("\n===== PHASE 17 : par zone =====")
    for nom, g in z.groupby("zone"):
        hc = int(g["canular"].sum())
        r = 100 * recall_score(g["canular"], g["pred"]) if hc > 0 else float("nan")
        p = 100 * precision_score(g["canular"], g["pred"], zero_division=0)
        print(f"  {nom:18} n={len(g):5d} canulars={hc:3d} prop={100*g['canular'].mean():.2f}% rappel={r:.0f}% précision={p:.1f}%")

    # ===== PHASE 18 : dérive de l'étiquette =====
    prop_an = (df.groupby("annee")["canular"].mean() * 100).dropna()
    plt.figure(figsize=(9, 4))
    plt.plot(prop_an.index, prop_an.values, marker="o")
    plt.xlabel("année d'observation"); plt.ylabel("% de canulars")
    plt.title("Proportion de canulars par année"); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig("canulars_par_annee.png", dpi=110)
    print("\n===== PHASE 18 : dérive =====")
    print("Courbe sauvegardée : canulars_par_annee.png")
    print(f"Train ancien -> test récent : rappel {100*recall_score(yte, proba>=SEUIL):.1f} % "
          f"| précision {100*precision_score(yte, proba>=SEUIL, zero_division=0):.2f} %")
    print(f"Indicateur 1 - taux flagué canular : {100*(proba>=SEUIL).mean():.2f} %")
    print(f"Indicateur 2 - proba moyenne        : {proba.mean():.3f}")
    print(f"Indicateur 3 - part pays manquant   : {100*df.loc[tst,'country_manque'].mean():.2f} %")