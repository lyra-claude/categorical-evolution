# Genetic Algorithms as Directed Containers

## The Basic GA

A genetic algorithm evolving a population over generations is a directed container.

```
S = PopulationShape                -- (population size, genome length, generation count)
P s = IndividualPositions s        -- each individual in each generation
o s = bestIndividual s             -- the fittest individual (the "answer")
down s (g, i) = subEvolution s g   -- the sub-evolution from generation g onward
(g, i) + (g', i') = (g + g', i')  -- individual i' in sub-evolution from g
                                      is individual i' at generation g + g' globally
```

### extract and duplicate

```
extract pop = pop ! bestIndividual     -- the fittest individual found
duplicate pop = at each (g, i),        -- the full evolutionary context at that point:
                replace individual       what generation, what fitness, what came before,
                with subEvolution(g)     what comes after
```

`extract` gives the answer: the best solution found.

`duplicate` gives the full evolutionary history at every point. For individual i at generation g, `duplicate` provides the entire sub-evolution from g onward — showing how that individual's descendants evolved, which ones survived selection, which were crossed over, which were mutated.

### Comonad laws

1. `extract . duplicate = id` — extracting the best from the duplicated structure recovers the original population. The evolutionary history is consistent with the final answer.

2. `fmap extract . duplicate = id` — extracting the best at every generation after duplicating recovers the original fitness trajectory.

3. `duplicate . duplicate = fmap duplicate . duplicate` — unfolding the evolutionary history twice is the same as unfolding and then unfolding each sub-evolution. Zooming into generation g's sub-evolution and then zooming into generation g' within that is the same as zooming directly into generation g + g'.

---

## Island Model GA

An island model has multiple sub-populations with periodic migration. This is where directed containers get interesting.

```
S = IslandModelShape               -- (num_islands, island_size, generations, topology)
P s = (Island, Generation, Index)  -- individual i on island k at generation g
o s = globalBest s                 -- best individual across all islands
down s (k, g, i) = subEvolution s k g  -- sub-evolution on island k from generation g
(k, g, i) + (k', g', i') = ...    -- embedding depends on migration topology
```

### The (+) operation encodes migration

Here is where the topology enters. The `(+)` operation — embedding positions from a sub-evolution into the global structure — depends on whether island k can receive individuals from island k'.

**No migration (none):**
```
(k, g, i) + (k, g', i') = (k, g + g', i')    -- positions stay on their island
(k, g, i) + (k', g', i') = undefined          -- k ≠ k': no cross-island embedding
```
Sub-evolutions are completely independent. `down` at island k gives a sub-evolution that is fully isolated. Maximum `down`-structure preserved. Maximum diversity.

**Ring topology:**
```
(k, g, i) + (k, g', i') = (k, g + g', i')              -- same island: direct
(k, g, i) + (k±1, g', i') = (k, g + g', migrated(i'))  -- neighbours only
(k, g, i) + (k', g', i') = undefined                     -- |k-k'| > 1: no path
```
Sub-evolutions on adjacent islands can exchange individuals at migration events. Partial `down`-structure preserved. High diversity.

**Fully connected:**
```
(k, g, i) + (k', g', i') = (k, g + g', migrated(i'))    -- any island to any island
```
Every sub-evolution can receive from every other. `down` at island k gives a sub-evolution that is influenced by all other islands. Minimal `down`-structure preserved. Low diversity — populations converge because every island's local evolution is contaminated by global mixing.

### Topology determines diversity: the directed container explanation

Diversity is a measure of how much independent structure exists in the population. In directed container terms, diversity is **how much `down`-structure the (+) operation preserves**.

- `none`: `(+)` is maximally restrictive — positions only embed within their own island. Each `down` gives a genuinely independent sub-evolution. High diversity.
- `ring`: `(+)` allows limited cross-embedding between neighbours. `down` gives mostly-independent sub-evolutions with some coupling. Medium-high diversity.
- `star`: `(+)` allows cross-embedding through a hub. `down` gives sub-evolutions coupled through one central island. Medium diversity.
- `fully_connected`: `(+)` allows unrestricted cross-embedding. `down` gives sub-evolutions that are all coupled to each other. Low diversity.

**The diversity ordering is the ordering of restrictiveness of `(+)`.**

This is Lyra's conjecture formulated as a statement about directed containers: the migration topology determines diversity because it determines the `(+)` operation, which determines how much independent `down`-structure exists.

---

## The Coevolutionary Case

Coevolution (multiple populations with fitness interdependence) adds a second layer:

```
S = CoevoShape                     -- (num_species, pop_sizes, interactions)
P s = (Species, Generation, Index)
o s = nashEquilibrium s            -- or Pareto front, or best-response tuple
down s (sp, g, i) = subCoevo s sp g
```

Now `down` at species sp gives the sub-evolution of that species — but fitness depends on the other species' state. The directed container structure captures this: `duplicate` at (sp, g, i) gives the full co-evolutionary context, including the state of competing species that determines fitness.

The comonad laws guarantee that the co-evolutionary context is consistent: zooming into species A's sub-evolution and then looking at species B's influence is the same as zooming into the full co-evolution and restricting to A's perspective.

---

## Side by Side

| | Single-Population GA | Island Model GA | Agent Pipeline |
|---|---|---|---|
| Shape | (pop_size, generations) | (num_islands, topology, ...) | (agents, ordering) |
| Positions | individuals at each gen | individuals on each island at each gen | outputs at each agent step |
| Root `o` | best individual | global best | published claim |
| `down` | sub-evolution from gen g | sub-evolution on island k from gen g | sub-task from agent a |
| `(+)` | generation embedding | depends on topology | depends on orchestrator |
| `extract` | the answer | the answer | the theorem |
| `duplicate` | evolutionary history | per-island evolutionary history | provenance chain |
| Diversity | N/A (one population) | determined by `(+)` | determined by orchestrator |

The migration topology for GAs, the DeFi aggregator for Ethereum, the Rules value for tax, and the meta-agent for the AI Mathematician are all the same thing: **the functor that determines the `(+)` operation on the directed container**.

Topology determines diversity. The orchestrator determines traceability. Same mathematics.
