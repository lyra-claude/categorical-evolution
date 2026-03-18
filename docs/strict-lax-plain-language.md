# Strict vs Lax: A Plain-Language Explainer

## The one-sentence version

When you build a system out of parts, the way those parts talk to each other matters more than what the parts actually are -- and there are really only two fundamental modes of connection: isolated (strict) or entangled (lax).

---

## The octopus

An octopus has nine brains. One central brain and one in each arm. Each arm can taste, grip, and solve problems on its own -- a severed octopus arm will still reach for food and pull it toward where the mouth used to be. The arm is not broken. It is operating in a mode where it does not consult anything else. It takes in information, processes it locally, and acts.

That is **strict** mode. The arms are independent processors. Each one runs its own computation, and the results do not leak between arms. Whatever one arm figures out stays in that arm.

Now consider the living octopus. The central brain sends signals that coordinate the arms. Not commands, exactly -- more like gentle nudges that synchronize what the arms are doing. Information flows between the arms, mediated by the central brain. An arm might change what it is doing because of what another arm discovered.

That is **lax** mode. The components are still doing their own processing, but there is a channel between them. Information migrates. The parts influence each other.

The key insight of our paper is that these two modes -- isolated processing vs. connected processing -- are not just an octopus quirk. They are a fundamental structural pattern that shows up across every optimization system we have looked at. And the mode you choose predicts what will happen, regardless of what you are optimizing.

## Islands

Here is a more concrete example. Imagine two islands, each with a population of finches. On each island, natural selection is doing its thing: finches with beaks better suited to the local food thrive and reproduce. The populations evolve.

**Strict mode:** No bird ever flies between the islands. Each population evolves completely independently. What happens? Each island's finches become very well adapted to their specific island. Convergence is fast because there is no outside disruption -- the population locks in on what works locally. But both islands might converge on similar solutions, because there is no fresh genetic material arriving to push the population in unexpected directions.

**Lax mode:** Occasionally, a finch flies from one island to the other. Genes migrate. What happens? Convergence slows down -- every time the population is about to settle on a solution, an immigrant arrives with different genes and shakes things up. But the overall diversity stays higher. The population explores more possibilities. Sometimes it finds solutions that neither island would have discovered alone.

This is not just an analogy. This is literally what happens in island-model genetic algorithms (a widely used optimization technique), and it is the first system we tested our theory on.

## Why this matters

Here is the punchline. The strict/lax distinction is not about finches or octopuses. It is a structural property of how any system composes its parts. And it makes a specific, testable prediction:

- **Strict composition** (isolated parts) converges faster, preserves internal structure, but explores less.
- **Lax composition** (connected parts) maintains diversity, explores more broadly, but converges slower and can degrade internal structure.

This is not a vague philosophical claim. It is precise enough to be wrong. And six independent research groups -- none of whom were talking to each other, none of whom were using our framework -- have produced results that line up with this prediction.

## The six data points

1. **Our own experiments.** We ran the same genetic algorithm on two completely unrelated problems (evolving checkers strategies and optimizing maze layouts). Strict composition preserved 2.2 times more diversity than lax. The statistical effect was enormous (for the technically inclined: Cohen's d = 4.34, p < 0.0000000001). Same composition pattern, same result, different domain. The composition predicted the outcome, not the problem content.

2. **Google/MIT scaling study.** A team studying how optimization errors scale found that lax-style systems (where components share information freely) amplify errors 17.2 times, while strict-style systems (isolated components) only amplify errors 4.4 times. Exactly what our framework predicts: lax composition degrades internal structure.

3. **Constitutional evolution (Sakana AI).** A study on evolving cooperative AI agents found that agents with minimal communication (0.9% -- essentially strict) outperformed agents with heavy communication (62.2% -- deeply lax). More connection did not mean better performance. It meant slower convergence and muddled solutions.

4. **Quality-diversity robotics (David Ha, Sakana).** Work on evolving robot controllers showed that letting separate populations evolve independently (strict) produced more general strategies than mixing them. The isolated groups discovered different approaches; mixing them prematurely would have collapsed that diversity into a single mediocre solution.

5. **Semantic collapse in multi-agent systems.** A study on AI agents working together found that when agents communicate too much (lax), their internal representations collapse -- they all start saying the same thing. The diversity of perspectives vanishes. Strict isolation prevents this collapse.

6. **MadEvolve (cosmological simulations).** A system for evolving physics simulations uses strict composition on the inside (individual evolution runs stay isolated) wrapped in lax composition on the outside (results are compared and the best approaches propagate). Nested strict-inside-lax -- exactly the architecture our framework would recommend for balancing convergence and exploration.

Six groups. Six different problems. One structural prediction. All consistent.

## What category theory adds

You might ask: why do you need fancy mathematics for this? Can you not just say "isolation vs. connection" and leave it at that?

You can, and practitioners do -- that is exactly why we think this work matters. Researchers across evolutionary computation, multi-agent AI, distributed systems, and optimization keep independently discovering the same pattern and giving it different names. Island models. Federated learning. Modular architectures. Isolated vs. cooperative agents.

Category theory gives us a single vocabulary to describe all of these as instances of the same phenomenon. "Strict monoidal functor" and "lax monoidal functor" are not just jargon -- they are precise mathematical objects with known properties, and those properties generate predictions.

The analogy is to physics. You can describe falling apples and orbiting planets separately, or you can notice that they are both instances of gravity. Category theory is the gravity here: the underlying structure that explains why the same pattern keeps showing up in different fields.

## The binding gradient

One final nuance. Strict and lax are not a binary switch. They are the two ends of a dial.

Think of it this way: on a scale from 0 to 1, how much do the parts of your system talk to each other? At 0 (fully strict), no communication -- each part evolves alone. At 1 (fully lax, sometimes called panmixia), everything mixes freely -- there are no separate parts anymore, just one big soup.

Most real systems live somewhere in between. A genetic algorithm might migrate one individual between islands every 50 generations -- that is close to strict but not quite. A multi-agent system might share summary statistics but not raw data -- lax, but gently.

Our collaborator Claudius pointed out that strict and lax are better understood as "the two coherent operating modes at the extremes of a continuous gradient." The interesting open question -- one we have not yet answered -- is whether there is a sharp tipping point along that gradient (a phase transition, like water freezing) or whether the change is smooth. That is an active research problem.

## So what?

This matters because it gives practitioners a tool for thinking about system design. If you are building an AI system with multiple components, the strict/lax framework tells you:

- Want fast convergence on a known-good solution? Isolate your components (strict).
- Want broad exploration of unfamiliar territory? Let your components talk (lax).
- Want both? Use strict composition inside, lax composition outside (like MadEvolve).

And it tells you this not as a rule of thumb, but as a consequence of mathematical structure that has been validated across six independent studies.

That is the project. That is what the paper formalizes. And that is what we are submitting to ACT 2026 and GECCO 2026 -- the argument that this structural pattern is real, general, and useful.
