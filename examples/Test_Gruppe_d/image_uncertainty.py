"""shapiq + probly mit VGG16: Welche Bildbereiche machen das Modell unsicher?

Alles in einer Datei: das Spiel (``UnsicherheitsSpiel``), die Berechnung mit shapiq und die Ausgabe
(``erklaere_unsicherheit``). Dropout, DropConnect und/oder Ensemble sind wählbar.

    from image_uncertainty import erklaere_unsicherheit
    erklaere_unsicherheit("bild.jpg", methode="beide", raster=4, samples=5, p=0.2)   # TU, AU und EU

mass="alle" (Standard) erklärt Gesamt- (TU), aleatorische (AU) und epistemische Unsicherheit (EU)
getrennt; mass="total" / "aleatoric" / "epistemic" erklärt nur eine davon.
interaktionen=True schaltet zusätzlich die Interaktionen zwischen Feldpaaren (k-SII, Ordnung 2) ein.

Ablauf (wie im Projektleitfaden):
    1. probly macht VGG16 unsicherheitsbewusst (Dropout, DropConnect oder Ensemble)
    2. Das Bild wird in raster x raster Felder geteilt, die Felder sind die Spieler
    3. Fehlende Felder werden ersetzt (mean, gray, black oder blur)
    4. probly bewertet jede Bildvariante: P(Zielklasse) und Unsicherheit (TU, AU und EU)
    5. shapiq berechnet daraus die Shapley-Werte (exakt bis 10 Felder, sonst Näherung mit Budget)

Schnelltest ohne VGG16-Download:  python image_uncertainty.py
"""

import copy
from itertools import combinations

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

import shapiq
from shapiq.approximator import KernelSHAP, KernelSHAPIQ
from probly.method import dropconnect, dropout, ensemble
from probly.representer import representer
from probly.quantification import quantify

MITTEL, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]     # ImageNet-Normalisierung
SPALTE = {"total": 1, "aleatoric": 2, "epistemic": 3}           # Spalte 0 ist P(Klasse)
METHODEN = ("dropout", "dropconnect", "ensemble")


# =====================================================================================
# 1. Das Spiel
# =====================================================================================
class UnsicherheitsSpiel(shapiq.Game):
    """Spieler = Felder eines Rasters. Wert einer Koalition = Unsicherheit bei sichtbaren Feldern.

    bild:      Pfad oder PIL-Bild
    modell:    torch-Klassifikator (z. B. VGG16, Bild (B,3,224,224) -> Logits) oder bei
               methode="ensemble" eine Liste trainierter Mitglieder
    methode:   "dropout", "dropconnect" oder "ensemble"
    raster:    Felder pro Zeile und Spalte (3 = 9 Spieler)
    samples:   Dropout: Anzahl Durchläufe. Ensemble: Anzahl Mitglieder (bei einem einzelnen Modell)
    p:         Dropout: Anteil ausgeschalteter Neuronen. DropConnect: Anteil ausgeschalteter Gewichte
    mass:      "total", "aleatoric" oder "epistemic"
    klasse:    Index der erklärten Klasse, None = wahrscheinlichste
    ersetzung: "mean", "gray", "black" oder "blur"
    stoerung:  Ensemble aus einem einzelnen Modell: Rauschen auf der letzten Schicht jeder Kopie
               (Platzhalter für trainierte Mitglieder; 0 = alle Kopien gleich, dann ist EU = 0)
    seed:      fester Seed für Dropout-Masken und Rauschen
    """

    def __init__(self, bild, modell, methode="dropout", raster=3, samples=5, p=0.2, mass="total",
                 klasse=None, ersetzung="mean", stoerung=0.0, seed=0, batch=16):
        if methode not in METHODEN:
            raise ValueError("methode muss 'dropout', 'dropconnect' oder 'ensemble' sein")
        if mass not in SPALTE:
            raise ValueError("mass muss 'total', 'aleatoric' oder 'epistemic' sein")
        mitglieder = list(modell) if isinstance(modell, (list, tuple)) else None
        if mitglieder is not None and methode != "ensemble":
            raise ValueError("Eine Liste von Modellen ist ein Ensemble: methode='ensemble'")
        self.methode, self.seed, self.batch, self.spalte = methode, seed, batch, SPALTE[mass]

        # Bild: Seite auf 256 skalieren, 224 x 224 aus der Mitte (wie VGG16 es erwartet)
        if not isinstance(bild, Image.Image):
            bild = Image.open(bild)
        bild = bild.convert("RGB")
        s = 256 / min(bild.size)
        bild = bild.resize((round(bild.width * s), round(bild.height * s)), Image.BILINEAR)
        links, oben = (bild.width - 224) // 2, (bild.height - 224) // 2
        self.bild = np.asarray(bild.crop((links, oben, links + 224, oben + 224)))

        # Felder: für jeden Pixel die Nummer seines Feldes (0 .. raster*raster-1)
        achse = np.minimum(np.arange(224) * raster // 224, raster - 1)
        self.felder = achse[:, None] * raster + achse[None, :]

        # Ersatzbild für verdeckte Felder
        if ersetzung == "mean":
            self.ersatz = np.broadcast_to(self.bild.reshape(-1, 3).mean(0).round().astype(np.uint8), self.bild.shape)
        elif ersetzung in ("gray", "grau"):
            self.ersatz = np.full_like(self.bild, 128)
        elif ersetzung == "black":
            self.ersatz = np.zeros_like(self.bild)
        elif ersetzung == "blur":
            self.ersatz = np.asarray(Image.fromarray(self.bild).filter(ImageFilter.GaussianBlur(224 / 32)))
        else:
            raise ValueError("ersetzung muss 'mean', 'gray', 'black' oder 'blur' sein")

        # probly macht das Modell unsicherheitsbewusst
        modelle = [m.eval() for m in (mitglieder or [modell])]
        self.modell = modelle[0]
        if methode == "dropout":
            gedroppt = dropout(modelle[0], p=p, predictor_type="logit_classifier")
            self.rep = representer(gedroppt, num_samples=samples)
        elif methode == "dropconnect":
            # Eigene Dropout-Schichten von VGG16 entfernen, sonst bleiben sie beim Ziehen der Stichproben aktiv
            gedroppt = dropconnect(_ohne_dropout(modelle[0]), p=p, predictor_type="logit_classifier")
            self.rep = representer(gedroppt, num_samples=samples)
        else:
            n = len(modelle) if mitglieder else samples
            ens = ensemble(modelle[0], num_members=n, reset_params=False, predictor_type="logit_classifier")
            for platz, mitglied in zip(ens, modelle * n):
                platz.load_state_dict(mitglied.state_dict())
            if stoerung and not mitglieder:
                gen = torch.Generator().manual_seed(seed)
                for kopie in ens:
                    letzte = [m for m in kopie.modules() if isinstance(m, torch.nn.Linear)][-1]
                    with torch.no_grad():
                        letzte.weight.add_(stoerung * letzte.weight.std() * torch.randn(letzte.weight.shape, generator=gen))
            self.rep = representer(ens.eval())

        # erklärte Klasse: die wahrscheinlichste des ungestörten Modells (oder vorgegeben)
        with torch.no_grad():
            self.wahrsch = torch.softmax(self.modell(self._tensor(self.bild[None]))[0], dim=0).numpy()
        self.klasse = int(self.wahrsch.argmax()) if klasse is None else klasse
        self._cache = {}                                    # Koalition -> [P, TU, AU, EU]
        super().__init__(n_players=raster * raster, normalize=False)

    def value_function(self, koalitionen):
        """Unsicherheit (mass) pro Koalition."""
        return self._werte(koalitionen)[:, self.spalte]

    def unsicherheit(self, koalitionen, art):
        """TU, AU oder EU ("total", "aleatoric", "epistemic") pro Koalition (aus demselben Zwischenspeicher)."""
        return self._werte(koalitionen)[:, SPALTE[art]]

    def class_probability(self, koalitionen):
        """Mittlere Wahrscheinlichkeit der erklärten Klasse pro Koalition (aus demselben Zwischenspeicher)."""
        return self._werte(koalitionen)[:, 0]

    def _werte(self, koalitionen):
        koalitionen = np.asarray(koalitionen, dtype=bool)
        keys = [z.tobytes() for z in koalitionen]
        neu = {}
        for i, key in enumerate(keys):
            if key not in self._cache and key not in neu:
                neu[key] = i
        if neu:                                             # jede Kombination geht nur EINMAL durch das Modell
            for key, zeile in zip(neu, self._rechne(koalitionen[list(neu.values())])):
                self._cache[key] = zeile
        return np.stack([self._cache[key] for key in keys])

    def _rechne(self, koalitionen):
        bilder = np.repeat(self.bild[None], len(koalitionen), axis=0)
        for i, z in enumerate(koalitionen):
            weg = ~z[self.felder]
            bilder[i][weg] = self.ersatz[weg]
        # Dropout: ein Bild pro Durchgang, damit jede Koalition dieselben Masken sieht.
        # DropConnect (eine Maske pro Gewichtsmatrix, für alle Bilder gleich) und Ensemble:
        # Batches fester Größe (aufgefüllt), damit die Werte nicht vom Batch abhängen.
        groesse = 1 if self.methode == "dropout" else self.batch
        zeilen = []
        for start in range(0, len(bilder), groesse):
            teil = bilder[start:start + groesse]
            voll = np.concatenate([teil, np.repeat(teil[-1:], groesse - len(teil), axis=0)])
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                torch.manual_seed(self.seed)
                sample = self.rep.represent(self._tensor(voll))
                q = quantify(sample)
                p = sample.sample_mean().probabilities[:, self.klasse]
                werte = torch.stack([p, q.total, q.aleatoric, q.epistemic], dim=1)
            zeilen.append(werte[:len(teil)].numpy())
        return np.concatenate(zeilen).astype(float)

    @staticmethod
    def _tensor(bilder):
        x = torch.as_tensor(np.array(bilder)).permute(0, 3, 1, 2).float() / 255
        return (x - torch.tensor(MITTEL).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)


# =====================================================================================
# 2. Berechnung mit shapiq und Ausgabe
# =====================================================================================
_cache_vgg = {}


def _lade_vgg():
    """Vortrainiertes VGG16 und die Klassennamen (wird nur einmal geladen)."""
    if "vgg" not in _cache_vgg:
        from torchvision.models import vgg16, VGG16_Weights
        gewichte = VGG16_Weights.IMAGENET1K_V1
        _cache_vgg["vgg"] = (vgg16(weights=gewichte).eval(), gewichte.meta["categories"])
    return _cache_vgg["vgg"]


def _ohne_dropout(modell):
    """Kopie des Modells, bei der alle nn.Dropout-Schichten durch nn.Identity ersetzt sind.

    VGG16 hat zwei eigene Dropout-Schichten (p = 0.5). probly schaltet beim Ziehen der Stichproben den
    Trainingsmodus an, und dann würden diese mitzufallen. Für reines DropConnect werden sie entfernt.
    """
    kopie = copy.deepcopy(modell)
    for eltern in kopie.modules():
        for name, kind in list(eltern.named_children()):
            if isinstance(kind, nn.Dropout):
                setattr(eltern, name, nn.Identity())
    return kopie


def _shapley(spiel, n_spieler, budget):
    """Shapley-Werte: exakt bei bis zu 10 Feldern, sonst KernelSHAP-Näherung mit Budget."""
    if n_spieler <= 10:
        sv = shapiq.ExactComputer(game=spiel, n_players=n_spieler)(index="SV", order=1)
        basis = sv.dict_values.get((), 0.0)
    else:
        sv = KernelSHAP(n=n_spieler, random_state=0).approximate(budget=budget, game=spiel)
        basis = sv.baseline_value
    return np.array([sv.dict_values.get((i,), 0.0) for i in range(n_spieler)]), basis


TITEL = {"total": "GESAMTUNSICHERHEIT (TU)", "aleatoric": "ALEATORISCHEN Unsicherheit (AU, Rauschen im Bild)",
         "epistemic": "EPISTEMISCHEN Unsicherheit (EU, fehlendes Wissen)"}
KURZ = {"total": "TU", "aleatoric": "AU", "epistemic": "EU"}


def erklaere_unsicherheit(bild, methode="dropout", raster=3, samples=5, p=0.2, mass="alle",
                          budget=1024, ziel=None, ersetzung="mean", seed=0, stoerung=0.1, plot=True, interaktionen=False,
                          top_paare=5):
    """Erklärt mit shapiq, welche Bildfelder VGG16 unsicher machen.

    bild:      Pfad zum Bild
    methode:   "dropout", "dropconnect", "ensemble", "beide" (Dropout und Ensemble)
               oder "alle" (Dropout, DropConnect und Ensemble)
    raster:    Felder pro Zeile und Spalte (3 = 9 Felder, 4 = 16 Felder)
    samples:   Dropout/DropConnect: Anzahl Durchläufe pro Bildvariante. Ensemble: Anzahl Mitglieder
    p:         Dropout: Anteil ausgeschalteter Neuronen. DropConnect: Anteil ausgeschalteter Gewichte (0.2 = 20 %)
    mass:      "alle" (Standard: TU, AU und EU getrennt erklären), "total", "aleatoric" oder "epistemic"
    budget:    Anzahl Kombinationen bei der Näherung (nur bei mehr als 10 Feldern)
    ziel:      Klassenname (Teilstring) oder None = wahrscheinlichste Klasse
    ersetzung: "mean", "gray" (oder "grau"), "black" oder "blur"
    seed:      fester Seed für Dropout-Masken und Ensemble-Rauschen
    stoerung:  Ensemble: Platzhalter-Rauschen der Mitglieder (siehe UnsicherheitsSpiel)
    interaktionen: False = aus (Standard), True = zusätzlich Interaktionen zwischen je zwei Feldern (k-SII).
               Bis 10 Felder exakt und ohne zusätzliche Modellrechnung, darüber Näherung mit dem budget.
    top_paare: Wie viele der stärksten Feldpaare ausgegeben und im Bild eingezeichnet werden (Standard 5)

    Gibt ein Dictionary {methode: {...}} zurück. Die Shapley-Werte pro Unsicherheitsart stehen unter
    "werte" (z. B. ergebnisse["dropout"]["werte"]["epistemic"]).
    Mit mass="alle" kostet die Zerlegung keine zusätzliche Modellrechnung: TU, AU und EU kommen aus
    denselben Auswertungen (Zwischenspeicher).
    """
    if methode not in METHODEN + ("beide", "alle"):
        raise ValueError("methode muss 'dropout', 'dropconnect', 'ensemble', 'beide' oder 'alle' sein")
    if mass != "alle" and mass not in SPALTE:
        raise ValueError("mass muss 'alle', 'total', 'aleatoric' oder 'epistemic' sein")
    methoden = {"beide": ["dropout", "ensemble"], "alle": list(METHODEN)}.get(methode, [methode])
    arten = list(SPALTE) if mass == "alle" else [mass]

    vgg, klassen = _lade_vgg()
    ziel_idx = None
    if ziel is not None:
        treffer = [i for i, k in enumerate(klassen) if ziel.lower() in k.lower()]
        if not treffer:
            raise ValueError(f"Keine Klasse mit '{ziel}' gefunden")
        ziel_idx = treffer[0]
    print("Verwendetes Bild:", bild)

    N = raster
    ergebnisse = {}
    for name in methoden:
        spiel = UnsicherheitsSpiel(bild, vgg, name, raster=N, samples=samples, p=p, mass=arten[0], klasse=ziel_idx,
                                   ersetzung=ersetzung, stoerung=stoerung if name == "ensemble" else 0.0, seed=seed)
        if name == methoden[0]:
            print("\nTop 5 von VGG16:")
            for i in np.argsort(spiel.wahrsch)[::-1][:5]:
                print(f"  {int(i):4d}  {klassen[i]:28s} {spiel.wahrsch[i]:.3f}")
            print(f"\nErklärte Klasse: {klassen[spiel.klasse]}  (P = {spiel.wahrsch[spiel.klasse]:.3f})")

        print(f"\n{'=' * 60}\nMethode: {name}   (Raster {N}x{N}, samples={samples}"
              + (f", p={p}" if name in ("dropout", "dropconnect") else "") + f", Unsicherheit={mass})\n{'=' * 60}")

        werte_p, basis_p = _shapley(spiel.class_probability, N * N, budget)     # der langsame Teil
        werte, basis = {}, {}
        for art in arten:                                                       # nutzt den Zwischenspeicher
            werte[art], basis[art] = _shapley(lambda z, a=art: spiel.unsicherheit(z, a), N * N, budget)
        voll = np.ones((1, N * N), dtype=bool)
        P_voll = spiel.class_probability(voll)[0]
        U_voll = {art: spiel.unsicherheit(voll, art)[0] for art in arten}

        print("\n=== Beitrag zur WAHRSCHEINLICHKEIT ===")
        print(np.round(werte_p.reshape(N, N), 3))
        print(f"Basis {basis_p:.4f} + Summe {werte_p.sum():.4f} = {basis_p + werte_p.sum():.4f}   (ganzes Bild: {P_voll:.4f})")
        for art in arten:
            titel = TITEL[art] if mass == "alle" else f"UNSICHERHEIT ({mass})"
            print(f"\n=== Beitrag zur {titel} ===")
            print(np.round(werte[art].reshape(N, N), 4))
            print(f"Basis {basis[art]:.5f} + Summe {werte[art].sum():.5f} = {basis[art] + werte[art].sum():.5f}   (ganzes Bild: {U_voll[art]:.5f})")

        print(f"\nWichtigstes Feld für die Klasse:           Feld {int(np.argmax(werte_p))}  ({werte_p.max():+.4f})")
        for art in arten:
            beschriftung = f"Feld mit größtem Beitrag zur {KURZ[art]}:" if mass == "alle" else "Feld mit größtem Beitrag zur Unsicherheit:"
            print(f"{beschriftung:43s}Feld {int(np.argmax(werte[art]))}  ({werte[art].max():+.5f})")
            print(f"Korrelation Wahrscheinlichkeit/{KURZ[art]}:".ljust(43) + f"{np.corrcoef(werte_p, werte[art])[0, 1]:+.3f}")
        if mass == "alle":
            au, eu = werte["aleatoric"], werte["epistemic"]
            print(f"\nGanzes Bild: TU {U_voll['total']:.5f} = AU {U_voll['aleatoric']:.5f} + EU {U_voll['epistemic']:.5f}")
            print("Felder, die vor allem AU treiben (Rauschen):        ",
                  [i for i in range(N * N) if au[i] > 0 and au[i] >= eu[i]])
            print("Felder, die vor allem EU treiben (fehlendes Wissen):",
                  [i for i in range(N * N) if eu[i] > 0 and eu[i] > au[i]])

        inter = {}
        if interaktionen:
            for art in arten:
                inter[art] = _interaktionen(lambda z, a=art: spiel.unsicherheit(z, a), N * N, budget)
                _drucke_interaktionen(art, mass, inter[art], U_voll[art], N, top_paare)

        if plot:
            if mass == "alle":
                _zeichne_alle(Image.fromarray(spiel.bild), klassen[spiel.klasse], name, N, werte_p, werte, P_voll, U_voll)
            else:
                _zeichne(Image.fromarray(spiel.bild), klassen[spiel.klasse], name, mass, N, werte_p, werte[mass],
                         P_voll, U_voll[mass])
            if interaktionen:
                _zeichne_interaktionen(Image.fromarray(spiel.bild), name, N, inter, top_paare)
        ergebnisse[name] = {"werte_p": werte_p, "basis_p": basis_p, "P_voll": P_voll, "werte": werte, "basis": basis,
                            "voll": U_voll, "werte_u": werte[arten[0]], "basis_u": basis[arten[0]],
                            "U_voll": U_voll[arten[0]], "interaktionen": inter, "spiel": spiel}

    for m1, m2 in combinations(methoden, 2):
        for art in arten:
            a, b = ergebnisse[m1]["werte"][art], ergebnisse[m2]["werte"][art]
            print(f"\nVergleich {m1.capitalize()} gegen {m2.capitalize()}: "
                  f"Korrelation der Beiträge zur {KURZ[art]} {np.corrcoef(a, b)[0, 1]:+.3f}")
    return ergebnisse


def _interaktionen(spiel, n_spieler, budget):
    """k-SII der Ordnung 2: Einzelbeitrag je Feld und Zusammenspiel je Feldpaar.

    Bis 10 Felder exakt (alle Kombinationen liegen schon im Zwischenspeicher), sonst KernelSHAPIQ mit Budget.
    Gibt (einzel, paare, basis) zurück; paare ist eine symmetrische Matrix (Diagonale 0).
    """
    if n_spieler <= 10:
        iv = shapiq.ExactComputer(game=spiel, n_players=n_spieler)(index="k-SII", order=2)
    else:
        iv = KernelSHAPIQ(n=n_spieler, max_order=2, index="k-SII", random_state=0).approximate(budget=budget, game=spiel)
    einzel = np.array([iv.dict_values.get((i,), 0.0) for i in range(n_spieler)])
    paare = np.zeros((n_spieler, n_spieler))
    for i in range(n_spieler):
        for j in range(i + 1, n_spieler):
            paare[i, j] = paare[j, i] = iv.dict_values.get((i, j), 0.0)
    return einzel, paare, iv.dict_values.get((), 0.0)


def _staerkste_paare(paare, anzahl=5):
    i, j = np.triu_indices(len(paare), 1)
    return [(int(i[k]), int(j[k]), paare[i[k], j[k]]) for k in np.argsort(-np.abs(paare[i, j]))[:anzahl]]


def _drucke_interaktionen(art, mass, ergebnis, voll, N, top_paare=5):
    einzel, paare, basis = ergebnis
    summe_paare = np.triu(paare, 1).sum()
    titel = TITEL[art] if mass == "alle" else f"UNSICHERHEIT ({mass})"
    print(f"\n=== Interaktionen (k-SII, Ordnung 2) zur {titel} ===")
    print("Einzelbeiträge der Felder (ohne Zusammenspiel):")
    print(np.round(einzel.reshape(N, N), 4))
    print(f"Basis {basis:.5f} + Einzel {einzel.sum():.5f} + Paare {summe_paare:.5f} "
          f"= {basis + einzel.sum() + summe_paare:.5f}   (ganzes Bild: {voll:.5f})")
    print(f"Stärkste {top_paare} Paare (Feld = Zeile*{N} + Spalte, Zeile und Spalte zählen ab 0):")
    for rang, (a, b, w) in enumerate(_staerkste_paare(paare, top_paare), 1):
        print(f"  {rang:2d}. Feld {a:2d} (Z{a // N}/S{a % N}) und Feld {b:2d} (Z{b // N}/S{b % N}): {w:+.5f}  "
              + ("(zusammen mehr Unsicherheit als einzeln)" if w > 0 else "(ähnliche Information, redundant)"))


def _zeichne_interaktionen(bild, name, N, inter, top_paare=5):
    """Oben: Matrix aller Feldpaare. Unten: die stärksten Paare als Linien im Bild (rot = verstärkt, blau = redundant)."""
    arten = list(inter)
    fig, axes = plt.subplots(2, len(arten), figsize=(5.5 * len(arten), 10), squeeze=False)
    for spalte, art in enumerate(arten):
        _, paare, _ = inter[art]
        g = max(np.abs(paare).max(), 1e-12)
        ax = axes[0, spalte]
        im = ax.imshow(paare, cmap="RdBu_r", vmin=-g, vmax=g)
        ax.set_title(f"Interaktionen {KURZ[art]} [{name}]"); ax.set_xlabel("Feld"); ax.set_ylabel("Feld")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax = axes[1, spalte]
        ax.imshow(bild); ax.axis("off")
        mitte = lambda f: ((f % N + 0.5) * 224 / N, (f // N + 0.5) * 224 / N)
        for a, b, w in _staerkste_paare(paare, top_paare):
            (x1, y1), (x2, y2) = mitte(a), mitte(b)
            ax.plot([x1, x2], [y1, y2], color="red" if w > 0 else "blue", lw=1 + 5 * abs(w) / g, alpha=0.8)
        ax.set_title(f"Stärkste Paare ({KURZ[art]})\nrot = zusammen mehr, blau = redundant", fontsize=10)
    plt.tight_layout(); plt.show()


def _karte(fig, ax, karte, titel, fmt, N, g):
    """Eine Beitragskarte (rot = erhöht, blau = senkt) mit den Werten in den Feldern."""
    im = ax.imshow(karte, cmap="RdBu_r", vmin=-g, vmax=g)
    for r in range(N):
        for c in range(N):
            ax.text(c, r, fmt.format(karte[r, c]), ha="center", va="center", fontsize=8 if N > 3 else 10)
    ax.set_title(titel); ax.axis("off"); fig.colorbar(im, ax=ax, fraction=0.046)


def _zeichne_alle(bild, klasse, name, N, werte_p, werte, P_voll, voll):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes[0, 0].imshow(bild); axes[0, 0].axis("off")
    axes[0, 0].set_title(f"{klasse}  [{name}]\nP = {P_voll:.2f}   TU = {voll['total']:.3f} = "
                         f"AU {voll['aleatoric']:.3f} + EU {voll['epistemic']:.3f}")
    _karte(fig, axes[0, 1], werte_p.reshape(N, N), "shapiq: Beitrag zur Wahrscheinlichkeit", "{:.2f}", N,
           np.abs(werte_p).max())
    g = max(np.abs(werte[a]).max() for a in SPALTE)                 # gleiche Farbskala: TU, AU, EU vergleichbar
    _karte(fig, axes[0, 2], werte["total"].reshape(N, N), "Beitrag zur TU (gesamt)", "{:.3f}", N, g)
    _karte(fig, axes[1, 0], werte["aleatoric"].reshape(N, N), "Beitrag zur AU (Rauschen im Bild)", "{:.3f}", N, g)
    _karte(fig, axes[1, 1], werte["epistemic"].reshape(N, N), "Beitrag zur EU (fehlendes Wissen)", "{:.3f}", N, g)

    ax, au, eu = axes[1, 2], werte["aleatoric"], werte["epistemic"]
    ax.scatter(au, eu, s=50)
    for i in range(N * N):
        ax.annotate(str(i), (au[i], eu[i]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Beitrag zur AU (Rauschen)"); ax.set_ylabel("Beitrag zur EU (fehlendes Wissen)")
    ax.set_title("Jedes Feld: Rauschen oder fehlendes Wissen?")
    plt.tight_layout(); plt.show()


def _zeichne(bild, klasse, name, mass, N, werte_p, werte_u, P_voll, U_voll):
    karte_p, karte_u = werte_p.reshape(N, N), werte_u.reshape(N, N)
    schrift = 8 if N > 3 else 10
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes[0, 0].imshow(bild); axes[0, 0].axis("off")
    axes[0, 0].set_title(f"{klasse}  [{name}]\nP = {P_voll:.2f}   Unsicherheit ({mass}) = {U_voll:.4f}")

    for ax, karte, titel, fmt in [(axes[0, 1], karte_p, "shapiq: Beitrag zur Wahrscheinlichkeit", "{:.2f}"),
                                  (axes[0, 2], karte_u, f"shapiq: Beitrag zur Unsicherheit ({mass})", "{:.3f}")]:
        g = np.abs(karte).max()
        im = ax.imshow(karte, cmap="RdBu_r", vmin=-g, vmax=g)
        for r in range(N):
            for c in range(N):
                ax.text(c, r, fmt.format(karte[r, c]), ha="center", va="center", fontsize=schrift)
        ax.set_title(titel); ax.axis("off"); fig.colorbar(im, ax=ax, fraction=0.046)

    for ax, karte, titel in [(axes[1, 0], karte_p, "Überlagerung: Wahrscheinlichkeit"),
                             (axes[1, 1], karte_u, "Überlagerung: Unsicherheit")]:
        g = np.abs(karte).max()
        ax.imshow(bild)
        ax.imshow(karte, cmap="RdBu_r", vmin=-g, vmax=g, alpha=0.5, extent=(0, 224, 224, 0), interpolation="nearest")
        ax.set_title(titel); ax.axis("off")

    ax = axes[1, 2]
    ax.scatter(werte_p, werte_u, s=50)
    for i in range(N * N):
        ax.annotate(str(i), (werte_p[i], werte_u[i]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Beitrag zur Wahrscheinlichkeit"); ax.set_ylabel("Beitrag zur Unsicherheit")
    ax.set_title("Jedes Feld: wichtig für die Klasse? treibt es die Unsicherheit?")
    plt.tight_layout(); plt.show()


# =====================================================================================
# 3. Schnelltest (Grundtests aus dem Projektleitfaden), ohne VGG16-Download
# =====================================================================================
def selbsttest():
    """Prüft TU = AU + EU, identische Mitglieder (EU = 0), Effizienz und Reihenfolge-Unabhängigkeit."""
    def netz(seed):
        torch.manual_seed(seed)
        return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten(),
                             nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 10)).eval()

    bild = Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 80, 3), dtype=np.uint8))
    alle = np.array([[i >> k & 1 for k in range(4)] for i in range(16)], dtype=bool)   # alle 16 Koalitionen
    for methode in METHODEN:
        w = {m: UnsicherheitsSpiel(bild, netz(0), methode, raster=2, samples=4, mass=m, stoerung=0.5)(alle) for m in SPALTE}
        assert np.allclose(w["total"], w["aleatoric"] + w["epistemic"], atol=1e-5), "TU != AU + EU"
        spiel = UnsicherheitsSpiel(bild, netz(0), methode, raster=2, samples=4, stoerung=0.5)
        sv = shapiq.ExactComputer(game=spiel, n_players=4)(index="SV", order=1)
        summe = sum(sv.dict_values.get((i,), 0.0) for i in range(4))
        assert np.isclose(sv.dict_values.get((), 0.0) + summe, spiel(alle[-1:])[0], atol=1e-6), "Effizienz verletzt"
        sv3 = {m: shapiq.ExactComputer(game=lambda z, m=m: spiel.unsicherheit(z, m), n_players=4)(index="SV", order=1)
               for m in SPALTE}
        w3 = {m: np.array([v.dict_values.get((i,), 0.0) for i in range(4)]) for m, v in sv3.items()}
        assert np.allclose(w3["total"], w3["aleatoric"] + w3["epistemic"], atol=1e-5), "Shapley TU != AU + EU"
        for m in SPALTE:                                    # Interaktionen: Basis + Einzel + Paare = ganzes Bild
            einzel, paare, basis = _interaktionen(lambda z, m=m: spiel.unsicherheit(z, m), 4, 0)
            ganz = spiel.unsicherheit(alle[-1:], m)[0]
            assert np.isclose(basis + einzel.sum() + np.triu(paare, 1).sum(), ganz, atol=1e-5), "k-SII Effizienz verletzt"
        reihenfolge = np.random.default_rng(0).permutation(16)
        assert np.allclose(spiel(alle[reihenfolge]), spiel(alle)[reihenfolge], atol=1e-6), "Wert hängt von der Reihenfolge ab"
        print(f"{methode}: TU = AU + EU (Werte und Shapley-Werte), Effizienz, Interaktionen und Reihenfolge-Unabhängigkeit ok")
    eu = UnsicherheitsSpiel(bild, netz(0), "ensemble", raster=2, samples=4, mass="epistemic", stoerung=0.0)(alle)
    assert np.allclose(eu, 0, atol=1e-6), "Identische Mitglieder müssen EU = 0 haben"
    print("identische Ensemble-Mitglieder: EU = 0 ok")
    # DropConnect mit p = 0 ändert nichts, auch wenn das Modell eigene Dropout-Schichten hat (werden entfernt)
    torch.manual_seed(0)
    mit_dropout = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten(),
                                nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 10)).eval()
    eu = UnsicherheitsSpiel(bild, mit_dropout, "dropconnect", raster=2, samples=4, p=0.0, mass="epistemic")(alle)
    assert np.allclose(eu, 0, atol=1e-6), "DropConnect mit p = 0 muss EU = 0 haben"
    print("DropConnect mit p = 0: EU = 0 ok (eigene Dropout-Schichten entfernt)")


if __name__ == "__main__":
    selbsttest()
