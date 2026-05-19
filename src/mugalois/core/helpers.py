from mugalois.core.types import Triple, TriplePattern, Environment
# c.f definition page 3 of docs/triples_pseudocodes.pdf

def seedsOf(term: str, env: Environment) -> set:
    """
    seedsOf(x, v) = v(x) if x is variable, otherwise {x}.


    Exemples :
      term = "?x",     env["?x"] = {"Mike", "Dustin"}  → {"Mike", "Dustin"}
      term = "?x",     env["?x"] = {}                  → {}  (pas de seeds)
      term = "st:Mike" (URI liée, pas variable)         → {"st:Mike"}
    """
    if term.startswith("?"):
        return env.get(term)
    else:
        return {term}


def updateEnv(env: Environment, 
              triples: set[Triple], 
              s: str, 
              o: str):
    """
    updateEnv(v, T, s, o) :
      - si s is variable → v(s) UNION projection s over T
      - si o is variable → v(o) UNION projection o over T

    Definition page 3 Triples Doc

    Exemple :
      T = {(Mike, :isFriendWith, Eleven), (Dustin, :isFriendWith, Max)}
      s = "?x", o = "?y"
      → v("?x") devient {"Mike", "Dustin"}
      → v("?y") devient {"Eleven", "Max"}
    """
    if s.startswith("?"):
        values_s = {t.s for t in triples}
        env.update(s, values_s) # makes union
    if o.startswith("?"):
        values_o = {t.o for t in triples}
        env.update(o, values_o) # makes union
