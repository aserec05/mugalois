"""
µ-Galois — Ground Truth Builder  v3.0
======================================
Phase 1 (INCHANGÉE) — Seed list canonique
  ~500 scientifiques de référence absolue dont le LLM connaît les faits.
  Prédicats enrichis : on remplace field/influenced (vides sur DBpedia)
  par doctoralAdvisor, doctoralStudent, employer, spouse — bien remplis.

Phase 2 (CORRIGÉE) — Complétion par batch sans ORDER BY
  On pagine sur dbo:Scientist avec OFFSET/LIMIT en évitant ORDER BY
  (trop coûteux pour le endpoint public → timeout → 0 résultats).
  Pour chaque sujet inconnu de la seed, on tire ses triples et on
  cumule jusqu'au quota TARGET_TRIPLES.

Sortie :
    ground_truth/
        subgraph.csv        — (subject, predicate, object, source)
        subgraph.ttl        — Turtle RDF
        ground_truth.json   — index T1/T2/T3
        stats.json          — métriques du corpus

Usage :
    pip install rdflib pandas
    python mugalois_ground_truth.py
"""

import json, time, sys
import pandas as pd
from pathlib import Path
from collections import defaultdict
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from rdflib import Graph, URIRef, Literal

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

ENDPOINT       = "https://dbpedia.org/sparql"
OUTPUT_DIR     = Path("ground_truth")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_TRIPLES = 5000
SLEEP_S        = 1.0

# Prédicats enrichis : field/influenced remplacés par des prédicats bien remplis
PREDICATES = [
    "http://dbpedia.org/ontology/birthPlace",
    "http://dbpedia.org/ontology/deathPlace",
    "http://dbpedia.org/ontology/award",
    "http://dbpedia.org/ontology/nationality",
    "http://dbpedia.org/ontology/almaMater",
    "http://dbpedia.org/ontology/knownFor",
    "http://dbpedia.org/ontology/doctoralAdvisor",
    "http://dbpedia.org/ontology/doctoralStudent",
    "http://dbpedia.org/ontology/employer",
    "http://dbpedia.org/ontology/spouse",
]

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 — SEED LIST CANONIQUE  (INCHANGÉE)
# ──────────────────────────────────────────────────────────────────────────────

SEED_SCIENTISTS = [
    # ── Physique ──────────────────────────────────────────────────────────────
    "Albert_Einstein", "Marie_Curie", "Richard_Feynman", "Niels_Bohr",
    "Werner_Heisenberg", "Erwin_Schrödinger", "Max_Planck", "Paul_Dirac",
    "Enrico_Fermi", "Ernest_Rutherford", "J._J._Thomson", "James_Clerk_Maxwell",
    "Isaac_Newton", "Galileo_Galilei", "Stephen_Hawking", "Peter_Higgs",
    "Murray_Gell-Mann", "Steven_Weinberg", "Sheldon_Glashow", "Abdus_Salam",
    "Roger_Penrose", "Kip_Thorne", "Vera_Rubin", "Lise_Meitner",
    "Wolfgang_Pauli", "Hendrik_Lorentz", "Heinrich_Hertz",
    # Nouveaux physique
    "Hans_Bethe", "Felix_Bloch", "Nils_Gustaf_Dalén", "Clinton_Davisson",
    "Peter_Debye", "Paul_Drude", "Freeman_Dyson", "Enrico_Fermi",
    "Vitaly_Ginzburg", "Roy_Glauber", "David_Gross", "Gerard_t%27Hooft",
    "Pyotr_Kapitsa", "Willis_Lamb", "Lev_Landau", "Max_von_Laue",
    "Leon_Lederman", "Tsung-Dao_Lee", "Gabriel_Lippmann", "Hendrik_Lorentz",
    "Albert_Michelson", "Robert_Millikan", "Ben_Mottelson", "Yoichiro_Nambu",
    "Walther_Nernst", "Lars_Onsager", "Martin_Perl", "Max_Born",
    "Julian_Schwinger", "Carlo_Rubbia", "Ernest_Walton", "Eugene_Wigner",
    "Chen-Ning_Yang", "Hideki_Yukawa", "Pieter_Zeeman", "Anton_Zeilinger",
    "Alain_Aspect", "John_Clauser", "Giorgio_Parisi",

    # ── Chimie ────────────────────────────────────────────────────────────────
    "Linus_Pauling", "Otto_Hahn", "Fritz_Haber", "Dorothy_Hodgkin",
    "Dmitri_Mendeleev", "Antoine_Lavoisier", "Robert_Boyle", "Humphry_Davy",
    "John_Dalton", "Svante_Arrhenius", "Jacobus_Henricus_van_%27t_Hoff",
    "Ilya_Prigogine", "Ahmed_Zewail", "Roald_Hoffmann",
    # Nouveaux chimie
    "Amedeo_Avogadro", "Adolf_von_Baeyer", "Emil_Fischer", "Paul_Flory",
    "Herbert_C._Brown", "Elias_James_Corey", "Marie_Curie",
    "Peter_Debye", "Johann_Deisenhofer", "Alan_Heeger",
    "Gerhard_Herzberg", "Dudley_Herschbach", "Kenichi_Fukui",
    "George_Andrew_Olah", "Linus_Pauling", "William_Ramsay",
    "Frederick_Soddy", "Richard_Synge", "Todd_Baron",
    "Harold_Urey", "Vincent_du_Vigneaud", "Georg_Wittig",

    # ── Biologie / Médecine ───────────────────────────────────────────────────
    "Charles_Darwin", "Gregor_Mendel", "Louis_Pasteur", "Alexander_Fleming",
    "Francis_Crick", "James_Watson", "Rosalind_Franklin", "Frederick_Sanger",
    "Barbara_McClintock", "Lynn_Margulis", "Carl_Woese",
    "Robert_Koch", "Joseph_Lister", "Edward_Jenner", "Paul_Ehrlich",
    "Christiaan_Barnard", "Elizabeth_Blackburn", "Craig_Venter",
    "Jennifer_Doudna", "Emmanuelle_Charpentier",
    # Nouveaux bio/médecine
    "Sydney_Brenner", "Michael_Brown_(biochemist)", "Baruj_Benacerraf",
    "Günter_Blobel", "Paul_Boyer", "Herbert_Boyer",
    "Michael_S._Brown", "Arvid_Carlsson", "George_Wald",
    "Max_Delbrück", "Christian_de_Duve", "Renato_Dulbecco",
    "Gertrude_Elion", "Gerald_Edelman", "John_Franklin_Enders",
    "Joseph_Goldstein", "Roger_Guillemin", "Alfred_Gilman",
    "Leland_Hartwell", "Walter_Hess", "David_Hubel",
    "François_Jacob", "Niels_Kaj_Jerne", "Eric_Kandel",
    "Arthur_Kornberg", "Hans_Krebs", "Joshua_Lederberg",
    "Rita_Levi-Montalcini", "André_Lwoff", "Salvador_Luria",
    "Peter_Medawar", "César_Milstein", "Jacques_Monod",
    "Thomas_Morgan", "Hermann_Muller", "Marshall_Nirenberg",
    "Severo_Ochoa", "Luc_Montagnier", "Françoise_Barré-Sinoussi",
    "Ivan_Pavlov", "Wilder_Penfield", "Max_Perutz",
    "Rosalyn_Yalow", "Andrew_Huxley", "Alan_Lloyd_Hodgkin",

    # ── Mathématiques ─────────────────────────────────────────────────────────
    "Carl_Friedrich_Gauss", "Leonhard_Euler", "Henri_Poincaré",
    "David_Hilbert", "John_von_Neumann", "Kurt_Gödel", "Alan_Turing",
    "Andrew_Wiles", "Grigori_Perelman", "Terence_Tao", "Maryam_Mirzakhani",
    "Srinivasa_Ramanujan", "Emmy_Noether", "Georg_Cantor",
    "Évariste_Galois", "Bernhard_Riemann", "Pierre-Simon_Laplace",
    "Blaise_Pascal", "Pierre_de_Fermat", "René_Descartes",
    # Nouveaux maths
    "Michael_Atiyah", "John_Tate", "Paul_Erdős", "Alexander_Grothendieck",
    "Laurent_Schwartz", "Jean-Pierre_Serre", "Jacques_Tits",
    "John_Milnor", "Stephen_Smale", "Paul_Cohen_(mathematician)",
    "William_Thurston", "Simon_Donaldson", "Vaughan_Jones",
    "Edward_Witten", "Maxim_Kontsevich", "Richard_Borcherds",
    "Timothy_Gowers", "Laurent_Lafforgue", "Wendelin_Werner",
    "Elon_Lindenstrauss", "Ngô_Bảo_Châu", "Cédric_Villani",
    "Artur_Avila", "Martin_Hairer", "Alessio_Figalli",
    "Peter_Scholze", "Akshay_Venkatesh", "Caucher_Birkar",

    # ── Informatique / IA ─────────────────────────────────────────────────────
    "Tim_Berners-Lee", "Donald_Knuth", "Edsger_W._Dijkstra",
    "John_McCarthy", "Marvin_Minsky", "Claude_Shannon", "Norbert_Wiener",
    "Yann_LeCun", "Geoffrey_Hinton", "Yoshua_Bengio", "Judea_Pearl",
    "Leslie_Lamport", "Barbara_Liskov", "Frances_Allen",
    "Dennis_Ritchie", "Ken_Thompson", "Linus_Torvalds",
    "Vint_Cerf", "Robert_E._Kahn",
    # Nouveaux informatique
    "John_Backus", "Edgar_Codd", "Ole-Johan_Dahl", "Niklaus_Wirth",
    "C._A._R._Hoare", "Ivan_Sutherland", "Butler_Lampson",
    "David_Patterson_(computer_scientist)", "John_L._Hennessy",
    "Silvio_Micali", "Shafi_Goldwasser", "Manuel_Blum",
    "Whitfield_Diffie", "Martin_Hellman", "Ron_Rivest",
    "Adi_Shamir", "Leonard_Adleman", "Michael_O._Rabin",
    "Dana_Scott", "Robin_Milner", "Amir_Pnueli",
    "Edmund_M._Clarke", "E._Allen_Emerson", "Joseph_Sifakis",
    "Andrew_Yao", "Turing_Award",

    # ── Astronomie / Cosmologie ───────────────────────────────────────────────
    "Nicolaus_Copernicus", "Johannes_Kepler", "Tycho_Brahe",
    "Edwin_Hubble", "Carl_Sagan", "Neil_deGrasse_Tyson",
    "Subrahmanyan_Chandrasekhar", "Fred_Hoyle", "Cecilia_Payne-Gaposchkin",
    "Annie_Jump_Cannon", "Henrietta_Swan_Leavitt",
    # Nouveaux astronomie
    "Saul_Perlmutter", "Brian_Schmidt", "Adam_Riess",
    "Russell_Hulse", "Joseph_Hooton_Taylor_Jr.", "Jocelyn_Bell_Burnell",
    "Andrea_Ghez", "Reinhard_Genzel", "Michel_Mayor",
    "Didier_Queloz", "George_Smoot", "John_C._Mather",
    "William_Alfred_Fowler", "Antony_Hewish", "Martin_Ryle",
    "Arno_Penzias", "Robert_Woodrow_Wilson", "Riccardo_Giacconi",
    "Raymond_Davis_Jr.", "Masatoshi_Koshiba",

    # ── Ingénierie / Inventeurs ───────────────────────────────────────────────
    "Nikola_Tesla", "Thomas_Edison", "James_Watt", "Michael_Faraday",
    "Archimedes", "Leonardo_da_Vinci",
    # Nouveaux ingénierie
    "Charles_Babbage", "Ada_Lovelace", "George_Boole",
    "Gottfried_Wilhelm_Leibniz", "Wilhelm_Conrad_Röntgen",
    "Alessandro_Volta", "André-Marie_Ampère", "Georg_Ohm",
    "Lord_Kelvin", "James_Prescott_Joule", "Ludwig_Boltzmann",
]

# Déduplique la seed list (quelques noms peuvent apparaître deux fois)
SEED_SCIENTISTS = list(dict.fromkeys(SEED_SCIENTISTS))

# ──────────────────────────────────────────────────────────────────────────────
# SPARQL HELPER
# ──────────────────────────────────────────────────────────────────────────────

def sparql_query(query: str, retries: int = 3, delay: float = 3.0) -> list[dict]:
    params = urlencode({
        "query": query,
        "format": "application/sparql-results+json",
    }).encode()
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "MuGaloisGroundTruth/3.0",
    }
    for attempt in range(1, retries + 1):
        try:
            req = Request(ENDPOINT, data=params, headers=headers, method="POST")
            with urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            return [
                {k: v["value"] for k, v in row.items()}
                for row in data["results"]["bindings"]
            ]
        except (HTTPError, URLError) as e:
            code = getattr(e, "code", "?")
            print(f"    [WARN] tentative {attempt}/{retries} — {code}: {e}", file=sys.stderr)
            if attempt < retries:
                time.sleep(delay * attempt)
    return []

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 — EXTRACTION SEED LIST  (INCHANGÉE)
# ──────────────────────────────────────────────────────────────────────────────

def extract_seed_triples() -> list[dict]:
    """Extrait tous les triples des prédicats cibles pour les scientifiques de la seed list."""
    print("\n── Phase 1 : seed list canonique ──────────────────────────────────")
    triples = []
    pred_filter = ", ".join(f"<{p}>" for p in PREDICATES)

    for name in SEED_SCIENTISTS:
        uri = f"http://dbpedia.org/resource/{name}"
        query = f"""
        SELECT ?p ?o WHERE {{
            <{uri}> ?p ?o .
            FILTER(?p IN ({pred_filter}))
        }}
        """
        rows = sparql_query(query)
        for r in rows:
            triples.append({
                "subject":   uri,
                "predicate": r["p"],
                "object":    r["o"],
                "source":    "seed",
            })
        if rows:
            print(f"  ✓ {name:<48} {len(rows)} triples")
        time.sleep(0.3)

    print(f"\n  → {len(triples)} triples issus de la seed list ({len(SEED_SCIENTISTS)} scientifiques)\n")
    return triples

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 — COMPLÉTION PAR BATCH  (CORRIGÉE : pas de ORDER BY)
# On pagine directement sur dbo:Scientist avec OFFSET/LIMIT.
# Pas de ORDER BY → pas de timeout. On filtre en local par nombre de triples
# pour ne garder que des entités avec au moins 2 prédicats (= bien couvertes).
# ──────────────────────────────────────────────────────────────────────────────

def extract_popular_triples(already_collected: int, seed_subjects: set) -> list[dict]:
    needed = TARGET_TRIPLES - already_collected
    if needed <= 0:
        print("  Quota atteint avec la seed list uniquement.")
        return []

    print(f"── Phase 2 : complétion par batch ({needed} triples manquants) ──────")

    pred_filter = ", ".join(f"<{p}>" for p in PREDICATES)
    triples     = []
    offset      = 0
    batch_size  = 100   # sujets par page — raisonnable pour le endpoint public
    max_empty   = 5     # arrêt si 5 pages consécutives sans résultat
    empty_count = 0

    while len(triples) < needed:
        # Récupère un batch de sujets Scientist
        q_subjects = f"""
        SELECT DISTINCT ?s WHERE {{
            ?s a <http://dbpedia.org/ontology/Scientist> .
            FILTER(STRSTARTS(STR(?s), "http://dbpedia.org/resource/"))
        }}
        LIMIT {batch_size} OFFSET {offset}
        """
        subjects = sparql_query(q_subjects)

        if not subjects:
            empty_count += 1
            print(f"  [page {offset//batch_size}] vide ({empty_count}/{max_empty})")
            if empty_count >= max_empty:
                print("  Trop de pages vides — fin de la complétion.")
                break
            offset += batch_size
            time.sleep(SLEEP_S)
            continue

        empty_count = 0

        for row in subjects:
            s = row["s"]
            if s in seed_subjects:
                continue

            q_triples = f"""
            SELECT ?p ?o WHERE {{
                <{s}> ?p ?o .
                FILTER(?p IN ({pred_filter}))
            }}
            """
            rows = sparql_query(q_triples)

            # On ne garde que les entités avec au moins 2 triples
            # (proxy de couverture — entité bien documentée)
            if len(rows) >= 2:
                for r in rows:
                    triples.append({
                        "subject":   s,
                        "predicate": r["p"],
                        "object":    r["o"],
                        "source":    "popular",
                    })
                label = s.split("/")[-1]
                total = already_collected + len(triples)
                print(f"  + {label:<50} {len(rows):>3} triples  (cumul: {total})")

            seed_subjects.add(s)   # évite de re-traiter ce sujet
            time.sleep(0.25)

            if len(triples) >= needed:
                break

        offset += batch_size
        time.sleep(SLEEP_S)

    print(f"\n  → {len(triples)} triples ajoutés (phase 2)\n")
    return triples

# ──────────────────────────────────────────────────────────────────────────────
# SAUVEGARDE
# ──────────────────────────────────────────────────────────────────────────────

def save_outputs(triples: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(triples).drop_duplicates(subset=["subject", "predicate", "object"])

    csv_path = OUTPUT_DIR / "subgraph.csv"
    df.to_csv(csv_path, index=False)
    print(f"[✓] CSV     → {csv_path}  ({len(df)} triples)")

    g = Graph()
    for _, row in df.iterrows():
        s = URIRef(row["subject"])
        p = URIRef(row["predicate"])
        o = URIRef(row["object"]) if str(row["object"]).startswith("http") else Literal(row["object"])
        g.add((s, p, o))
    ttl_path = OUTPUT_DIR / "subgraph.ttl"
    g.serialize(destination=str(ttl_path), format="turtle")
    print(f"[✓] Turtle  → {ttl_path}")

    return df

# ──────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH INDEX
# ──────────────────────────────────────────────────────────────────────────────

def build_ground_truth(df: pd.DataFrame) -> dict:
    gt = {"T1": defaultdict(list), "T2": defaultdict(list), "T3": defaultdict(list)}

    for _, row in df.iterrows():
        s, p, o = row["subject"], row["predicate"], row["object"]
        gt["T1"][f"{s}|{p}"].append(o)
        gt["T2"][f"{p}|{o}"].append(s)
        gt["T3"][p].append([s, o])

    out = {
        "T1": dict(gt["T1"]),
        "T2": dict(gt["T2"]),
        "T3": {k: v for k, v in gt["T3"].items()},
    }

    gt_path = OUTPUT_DIR / "ground_truth.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[✓] GT JSON → {gt_path}")
    print(f"    T1 clés: {len(out['T1'])}  |  T2 clés: {len(out['T2'])}  |  T3 prédicats: {len(out['T3'])}")
    return out

# ──────────────────────────────────────────────────────────────────────────────
# STATS
# ──────────────────────────────────────────────────────────────────────────────

def save_stats(df: pd.DataFrame):
    stats = {
        "total_triples":     int(len(df)),
        "unique_subjects":   int(df["subject"].nunique()),
        "unique_predicates": int(df["predicate"].nunique()),
        "unique_objects":    int(df["object"].nunique()),
        "triples_per_predicate": (
            df.groupby("predicate").size()
              .rename(lambda x: x.split("/")[-1])
              .to_dict()
        ),
        "triples_per_source": df["source"].value_counts().to_dict(),
    }
    stats_path = OUTPUT_DIR / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[✓] Stats   → {stats_path}")

    print("\n── Répartition par prédicat ────────────────────────────────────────")
    for pred, count in stats["triples_per_predicate"].items():
        bar = "█" * (count // 15)
        print(f"  {pred:<25} {count:>5}  {bar}")

# ──────────────────────────────────────────────────────────────────────────────
# ÉVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(predicted: list, ground_truth: list, query_id: str = "") -> dict:
    pred_set = set(map(tuple, predicted)) if predicted and isinstance(predicted[0], (list, tuple)) \
               else set(predicted)
    gt_set   = set(map(tuple, ground_truth)) if ground_truth and isinstance(ground_truth[0], (list, tuple)) \
               else set(ground_truth)

    tp        = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall    = tp / len(gt_set)   if gt_set   else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "query_id":  query_id,
        "TP":        tp,
        "predicted": len(pred_set),
        "GT":        len(gt_set),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         µ-Galois — Ground Truth Builder  v3.0                   ║")
    print(f"║  Seed list : {len(SEED_SCIENTISTS)} scientifiques | Prédicats : {len(PREDICATES)}              ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Phase 1 — INCHANGÉE
    seed_triples  = extract_seed_triples()
    seed_subjects = {t["subject"] for t in seed_triples}

    # Phase 2 — Complétion robuste
    popular_triples = extract_popular_triples(len(seed_triples), seed_subjects)

    all_triples = seed_triples + popular_triples

    if not all_triples:
        print("\n[ERREUR] Aucun triple. Vérifiez la connectivité vers DBpedia.", file=sys.stderr)
        sys.exit(1)

    print(f"\n══ Total brut : {len(all_triples)} triples ══\n")

    df = save_outputs(all_triples)
    print(f"\n══ Après déduplication : {len(df)} triples ══\n")

    build_ground_truth(df)
    save_stats(df)

    print(f"\n✓ Ground truth prêt dans : {OUTPUT_DIR.resolve()}/")
    print("""
Utilisation :
    from mugalois_ground_truth import evaluate
    import json

    gt = json.load(open("ground_truth/ground_truth.json"))

    pred_key = "http://dbpedia.org/ontology/birthPlace"
    predicted_triples = my_strategy.run(pred_key)   # sortie réelle de µ-Galois

    metrics = evaluate(predicted_triples, gt["T3"][pred_key], query_id="T3|birthPlace")
    print(metrics)
""")

if __name__ == "__main__":
    main()