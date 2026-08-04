---
pageClass: plain-doc
---

# Ontology

Ontology spans two threads: the metaphysical tradition in philosophy that asks "what exists," and the knowledge-representation engineering in computer science that deals in "an explicit specification of a conceptualization." This plan unfolds as a two-branched knowledge tree: work through every chapter of the corresponding textbooks, and finish writing every entry listed here.

## Topic Plan

<ProgressGrid cat="advanced/ontology" />

### Part I · Philosophical Origins: The Question of Being

- [ ] Ontology's question of concern: being qua being
- [ ] Aristotle's *Categories*: the ten categories and substance
- [ ] Primary and secondary substance: individuals, species, and genera
- [ ] The four causes: material, formal, efficient, and final
- [ ] The medieval problem of universals: realism and nominalism
- [ ] Thomas Aquinas: the distinction between being (*esse*) and essence
- [ ] Ockham's razor: do not multiply entities beyond necessity
- [ ] Descartes: the cogito and mind–body dualism
- [ ] Spinoza: substance monism and modes
- [ ] Leibniz: monadology and pre-established harmony
- [ ] Hume: bundle theory and the dissolution of the idea of substance
- [ ] Kant's *Critique of Pure Reason*: transcendental deduction and the system of categories
- [ ] Things-in-themselves and appearances: the limits of ontology

### Part II · The Analytic Tradition: From Frege to Quine

- [ ] Frege's *Begriffsschrift*: the invention of functions and quantifiers
- [ ] Sense and reference: the semantics of proper names
- [ ] Russell's theory of descriptions: dissolving the grammatical illusion of "existence"
- [ ] Logical atomism: the world as the totality of facts
- [ ] Carnap: linguistic frameworks and the internal/external distinction in ontology
- [ ] Quine's "On What There Is": to be is to be the value of a bound variable
- [ ] Ontological commitment: inventorying the entities a theory is committed to
- [ ] Naturalized ontology: philosophy following science
- [ ] Meinong's theory of objects and the problem of nonexistent things

### Part III · Contemporary Issues: Modality, Parts, and Supervenience

- [ ] Foundations of modal logic: possible-worlds semantics
- [ ] Kripke's *Naming and Necessity*: rigid designators and essentialism
- [ ] Lewis's modal realism: do possible worlds really exist
- [ ] Mereology: the formal theory of parts and wholes
- [ ] Supervenience: mental properties and physical properties
- [ ] Persistence in time: endurantism vs perdurantism
- [ ] Metaphysical grounding: dependence deeper than causation
- [ ] The ontological status of fictional objects

### Part IV · Foundations of Knowledge Representation

- [ ] The five roles of knowledge representation (Davis et al.): what KR is
- [ ] Semantic networks: nodes, arcs, and inheritance inference
- [ ] The semantic crisis of semantic networks: what links actually mean
- [ ] Frame systems: slots, facets, and defaults
- [ ] Default reasoning and nonmonotonicity in inheritance networks
- [ ] Production rule systems: forward chaining and backward chaining
- [ ] First-order logic representation: predicates, quantifiers, and knowledge bases
- [ ] The Frame Problem: representing a changing world
- [ ] Commonsense representation: default logic and circumscription

### Part V · Description Logics

- [ ] From KL-ONE to description logics: the lineage of terminological systems
- [ ] TBox and ABox: terminological knowledge and world assertions
- [ ] The ALC language: concept constructors and semantics
- [ ] Extending the ALC family: transitive roles, inverse roles, number restrictions (SROIQ)
- [ ] The open-world assumption (OWA) and the closed-world assumption (CWA)
- [ ] Tableau algorithms: deciding concept satisfiability
- [ ] The complexity of description logics: from PTIME to N2EXPTIME
- [ ] Classification and instance retrieval reasoning services

### Part VI · The Semantic Web Technology Stack

- [ ] The Semantic Web vision: Berners-Lee's layer cake
- [ ] The RDF data model: triples, IRIs, and blank nodes
- [ ] RDF syntaxes: Turtle, RDF/XML, and JSON-LD
- [ ] RDFS: classes, subclasses, properties, domains, and ranges
- [ ] OWL 2 constructors: equivalence, disjointness, property chains, and cardinality constraints
- [ ] OWL 2 Profiles: use cases for EL, QL, and RL
- [ ] The SPARQL query language: graph patterns, OPTIONAL, and FILTER
- [ ] SPARQL 1.1: aggregates, subqueries, and federated queries
- [ ] Reasoner practice: Pellet, HermiT, FaCT++
- [ ] SHACL: shape constraints and validation for RDF data

### Part VII · Ontology Engineering

- [ ] Ontology engineering overview: the lifecycle from requirements to deployment
- [ ] The METHONTOLOGY methodology: specification, conceptualization, formalization
- [ ] Competency questions: scoping with questions
- [ ] The NeOn methodology: scenario-driven development of ontology networks
- [ ] Ontology design patterns (ODPs): reusable modeling solutions
- [ ] The point of upper ontologies: why share a top level
- [ ] BFO (Basic Formal Ontology): a realist upper ontology
- [ ] DOLCE: a cognitive descriptive ontology
- [ ] SUMO: the Suggested Upper Merged Ontology
- [ ] Ontology matching and alignment: automatically discovering equivalence relations

### Part VIII · Knowledge Graphs

- [ ] The conceptual history of knowledge graphs: from the Semantic Web to the Google Knowledge Graph
- [ ] Knowledge graph construction overview: top-down and bottom-up
- [ ] Entity extraction: rule-based, statistical, and neural methods for named entity recognition
- [ ] Relation extraction: distant supervision and joint extraction
- [ ] Knowledge fusion: entity alignment and entity resolution
- [ ] Technical routes for schema alignment and instance alignment
- [ ] Graph databases: Neo4j and the property graph model
- [ ] RDF triple stores: Jena, Virtuoso, and GraphDB
- [ ] The Cypher query language: pattern-matching graph queries
- [ ] Property graphs vs RDF: comparing two paradigms and their interoperability
- [ ] Knowledge representation learning: TransE-family embedding models

### Part IX · Knowledge Graphs × LLMs

- [ ] Knowledge graphs in the LLM era: complementary or substitutive
- [ ] GraphRAG: retrieval-augmented generation over graph structure
- [ ] From text to graph: LLM-driven knowledge extraction pipelines
- [ ] Ontology-constrained generation: using a schema to constrain LLM output
- [ ] Validating structured output: JSON Schema, SHACL, and an LLM validation loop
- [ ] An agent's world model: the graph as long-term memory
- [ ] Commonsense reasoning: ConceptNet, ATOMIC, and LLM commonsense
- [ ] Neuro-symbolic integration: the junction of symbolic reasoning and statistical learning
- [ ] Hallucination suppression: knowledge graphs as factual anchors

### Part X · An Introduction to Category Theory (A Programmer's Perspective)

- [ ] Why programmers should learn category theory: three leaps of abstraction
- [ ] The definition of a category: objects, morphisms, and composition
- [ ] The categories Set and Hask
- [ ] A monoid is a single-object category
- [ ] Products and coproducts: the first appearance of universal properties
- [ ] Initial objects and terminal objects
- [ ] Functors: mappings between categories
- [ ] Natural transformations: mappings between functors
- [ ] A monad is a monoid in the category of endofunctors
- [ ] Adjoint functors: the most universal universal construction
- [ ] The Yoneda lemma: an object is determined by its network of relations
- [ ] Category theory as ontology: "being" from a structuralist perspective

> When the writing is complete: create `xxx.md` in this directory, then change the corresponding entry above to `- [x] [标题](./xxx)`.
