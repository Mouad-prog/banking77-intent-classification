from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig_dir = Path("figures")
fig_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis("off")

def box(x, y, title, body, w=2.25, h=1.15):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.12,rounding_size=0.12",
        linewidth=1.5,
        facecolor="white",
        edgecolor="black"
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h*0.68, title, ha="center", va="center", fontsize=10.5, weight="bold")
    ax.text(x + w/2, y + h*0.32, body, ha="center", va="center", fontsize=8.7)
    return x, y, w, h

def arrow(a, b):
    ax.add_patch(FancyArrowPatch(
        a, b,
        arrowstyle="->",
        mutation_scale=17,
        linewidth=1.5
    ))

b1 = box(0.6, 4.2, "Dataset", "BANKING77\n13083 requêtes\n77 intentions")
b2 = box(3.2, 4.2, "EDA", "longueur requêtes\nrépartition intents\ntrain/test")
b3 = box(5.8, 4.2, "Prétraitement", "lowercase\ntokenization\nstopwords\nstemming")
b4 = box(8.4, 4.2, "Représentation", "BoW\nTF-IDF\nSBERT embeddings")

arrow((b1[0]+b1[2], b1[1]+b1[3]/2), (b2[0], b2[1]+b2[3]/2))
arrow((b2[0]+b2[2], b2[1]+b2[3]/2), (b3[0], b3[1]+b3[3]/2))
arrow((b3[0]+b3[2], b3[1]+b3[3]/2), (b4[0], b4[1]+b4[3]/2))

c1 = box(11.2, 5.55, "Baselines classiques", "BoW / TF-IDF\nLogReg / SVM")
c2 = box(11.2, 3.05, "Modèles sémantiques", "SBERT embeddings\nLogReg / SVM")

arrow((b4[0]+b4[2], b4[1]+b4[3]/2), (c1[0], c1[1]+c1[3]/2))
arrow((b4[0]+b4[2], b4[1]+b4[3]/2), (c2[0], c2[1]+c2[3]/2))

e1 = box(13.8, 5.55, "Évaluation", "accuracy\nmacro precision\nmacro recall\nmacro F1")
e2 = box(13.8, 3.05, "Analyse d'erreurs", "intentions difficiles\nconfusions\nexemples mal classés")

arrow((c1[0]+c1[2], c1[1]+c1[3]/2), (e1[0], e1[1]+e1[3]/2))
arrow((c2[0]+c2[2], c2[1]+c2[3]/2), (e2[0], e2[1]+e2[3]/2))

final = box(6.4, 1.0, "Résultats finaux", "meilleur modèle : SBERT + SVM\nmacro F1 = 92.76 %\namélioration vs BoW/TF-IDF", w=3.8, h=1.25)

arrow((e1[0]+1.1, e1[1]), (final[0]+final[2]*0.72, final[1]+final[3]))
arrow((e2[0]+1.1, e2[1]), (final[0]+final[2]*0.72, final[1]+final[3]))

ax.text(
    8, 7.35,
    "Architecture globale du projet : Intent Classification avec BANKING77",
    ha="center",
    va="center",
    fontsize=16,
    weight="bold"
)

ax.text(
    8, 0.35,
    "Objectif : comparer les représentations classiques BoW/TF-IDF avec les embeddings sémantiques SBERT",
    ha="center",
    va="center",
    fontsize=10
)

plt.tight_layout()
plt.savefig(fig_dir / "10_project_architecture.png", dpi=300, bbox_inches="tight")
plt.close()
