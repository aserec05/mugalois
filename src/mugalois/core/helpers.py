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
    if s.startswith("?"):
        values_s = {str(t.s) for t in triples}  # cast ici
        env.update(s, values_s)
    if o.startswith("?"):
        values_o = {str(t.o) for t in triples}  # cast ici
        env.update(o, values_o)