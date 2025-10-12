# Pharmacology Knowledge Graph

The **Pharmacology Knowledge Graph** agent analyzes pharmacology publications and constructs structured knowledge graphs representing drugs, targets, diseases, mechanisms, interactions, and their relationships.

## Overview

The pharmacology knowledge graph system provides:

- **Publication Parsing**: Extract structured content from PubMed abstracts, DOIs, PMIDs, or full texts
- **Entity Extraction**: Identify drugs, targets (proteins/genes), diseases, mechanisms, interactions, and outcomes
- **Relationship Extraction**: Discover relationships between entities (inhibits, treats, activates, etc.)
- **Knowledge Graph Construction**: Build structured graphs with nodes and edges
- **Interactive Visualization**: Create network visualizations with multiple layout algorithms
- **Multi-Format Export**: Export graphs in JSON, RDF/Turtle, Neo4j Cypher, NetworkX, GraphML
- **Graph Query**: Query for specific nodes, edges, paths, or subgraphs

## Quick Start

### Basic Workflow

```python
# 1. Parse a pharmacology publication
lobster: Parse the publication with PMID 12345678

# 2. Extract entities
lobster: Extract pharmacological entities from PMID 12345678

# 3. Extract relationships
lobster: Extract relationships between entities in PMID 12345678

# 4. Build knowledge graph
lobster: Build a knowledge graph from PMID 12345678

# 5. Visualize
lobster: Visualize the knowledge graph

# 6. Export
lobster: Export the knowledge graph as JSON
```

## Entity Types

The system extracts six types of pharmacological entities:

### 1. **Drug**
- Drugs, compounds, molecules, medications, pharmaceuticals
- Examples: aspirin, ibuprofen, paracetamol, atorvastatin

### 2. **Target**
- Proteins, genes, receptors, enzymes, kinases, channels
- Examples: COX-2, EGFR, TP53, dopamine receptor

### 3. **Disease**
- Diseases, disorders, syndromes, conditions, pathologies
- Examples: cardiovascular disease, cancer, diabetes, Alzheimer's

### 4. **Mechanism**
- Inhibition, activation, modulation, agonism, antagonism
- Examples: competitive inhibition, allosteric modulation

### 5. **Interaction**
- Drug-drug interactions, synergy, antagonism
- Examples: drug interaction, potentiation, contraindication

### 6. **Outcome**
- Efficacy, toxicity, adverse effects, clinical outcomes
- Examples: therapeutic effect, side effect, adverse reaction

## Relationship Types

The system identifies 12 types of relationships:

- **inhibits**: Drug inhibits target protein
- **activates**: Drug activates target protein
- **binds_to**: Drug binds to target
- **treats**: Drug treats disease
- **indicated_for**: Drug indicated for disease
- **contraindicated**: Drug contraindicated for condition
- **interacts_with**: Drug-drug interaction
- **metabolizes**: Enzyme metabolizes drug
- **targets**: Drug targets specific protein
- **associated_with**: Entity associated with outcome
- **causes**: Drug causes adverse effect
- **prevents**: Drug prevents disease

## Detailed Usage

### 1. Publication Parsing

Parse pharmacology publications from various sources:

```python
# Parse from full text
lobster: Parse this pharmacology text: "Aspirin inhibits COX-2 enzyme and treats cardiovascular disease..."

# Parse from PubMed ID
lobster: Parse publication PMID 12345678

# Parse from DOI
lobster: Parse publication with DOI 10.1234/journal.2024.001
```

**Options:**
- `publication_text`: Full text or abstract
- `publication_id`: PMID or DOI for tracking
- `source_type`: "full_text", "abstract", or "pdf"

### 2. Entity Extraction

Extract specific types of entities with confidence filtering:

```python
# Extract all entity types
lobster: Extract all pharmacological entities from PMID 12345678

# Extract specific entity types
lobster: Extract only drugs and targets from PMID 12345678

# Use high confidence threshold
lobster: Extract entities from PMID 12345678 with confidence threshold 0.8
```

**Options:**
- `entity_types`: List of types to extract (default: all)
- `confidence_threshold`: Minimum confidence (0.0-1.0, default: 0.5)

### 3. Relationship Extraction

Identify relationships between extracted entities:

```python
# Extract all relationships
lobster: Extract all relationships from PMID 12345678

# Extract specific relationship types
lobster: Extract "inhibits" and "treats" relationships from PMID 12345678
```

**Options:**
- `relationship_types`: List of types to extract (default: all)

### 4. Knowledge Graph Construction

Build structured knowledge graphs:

```python
# Build graph with default name
lobster: Build knowledge graph from PMID 12345678

# Build graph with custom name
lobster: Build knowledge graph named "aspirin_cox2_study" from PMID 12345678
```

**Graph Structure:**
- **Nodes**: Entities with properties (name, type, mentions)
- **Edges**: Relationships with properties (type, confidence, evidence)
- **Metadata**: Creation time, statistics, provenance

### 5. Visualization

Create interactive network visualizations:

```python
# Force-directed layout (default)
lobster: Visualize the knowledge graph

# Circular layout
lobster: Visualize knowledge graph with circular layout

# Hierarchical layout
lobster: Visualize knowledge graph with hierarchical layout by type
```

**Layout Options:**
- `force`: Force-directed spring layout
- `circular`: Nodes arranged in a circle
- `hierarchical`: Nodes arranged by type in layers

**Customization:**
- `node_size_by`: Attribute for node size ("mentions", "degree")
- `color_by`: Attribute for node color ("type", "confidence")

### 6. Export

Export knowledge graphs in multiple formats:

```python
# JSON (default)
lobster: Export knowledge graph as JSON

# RDF/Turtle for semantic web
lobster: Export knowledge graph as RDF

# Neo4j Cypher queries
lobster: Export knowledge graph for Neo4j

# NetworkX edge list
lobster: Export knowledge graph for NetworkX

# GraphML for Gephi/Cytoscape
lobster: Export knowledge graph as GraphML
```

**Export Formats:**

| Format | Description | Use Case |
|--------|-------------|----------|
| JSON | Standard JSON | Web applications, general use |
| RDF | RDF/Turtle | Semantic web, triple stores |
| Neo4j | Cypher queries | Neo4j graph database |
| NetworkX | Edge list | Python graph analysis |
| GraphML | XML format | Gephi, Cytoscape, visualization tools |

### 7. Query

Query knowledge graphs for specific patterns:

```python
# Query nodes
lobster: Find all drugs in the knowledge graph

# Query with filters
lobster: Find nodes of type "target" in the knowledge graph

# Query edges
lobster: Find all "inhibits" relationships in the knowledge graph

# Query paths
lobster: Find paths between aspirin and COX-2 in the knowledge graph

# Query subgraphs
lobster: Extract subgraph of all drug-target interactions
```

**Query Types:**
- `node`: Find nodes matching filters
- `edge`: Find edges matching filters
- `path`: Find paths between two nodes
- `subgraph`: Extract subgraph matching criteria

**Filter Examples:**
```python
# Node filters
{"type": "drug"}
{"name": "aspirin"}
{"type": "target", "mentions": 3}

# Edge filters
{"relation_type": "inhibits"}
{"confidence": 0.9}

# Path filters
{"source": "aspirin", "target": "COX-2", "max_length": 3}
```

## Advanced Workflows

### Multi-Publication Knowledge Graph

Combine multiple publications into a single knowledge graph:

```python
# Parse multiple publications
lobster: Parse publication PMID 11111111
lobster: Parse publication PMID 22222222
lobster: Parse publication PMID 33333333

# Extract entities from all
lobster: Extract entities from PMID 11111111
lobster: Extract entities from PMID 22222222
lobster: Extract entities from PMID 33333333

# Build combined graph
# (Manually combine entities and relationships, or process separately)
```

### Drug Discovery Workflow

Focus on drug-target interactions for drug discovery:

```python
# 1. Parse drug discovery literature
lobster: Parse publications about kinase inhibitors

# 2. Extract drugs and targets only
lobster: Extract drugs and targets from the publications

# 3. Focus on inhibition relationships
lobster: Extract "inhibits" and "binds_to" relationships

# 4. Build and visualize drug-target network
lobster: Build knowledge graph
lobster: Visualize with force layout sized by mentions

# 5. Query for drug repurposing opportunities
lobster: Find all drugs that target protein X
```

### Adverse Effect Analysis

Analyze drug side effects and interactions:

```python
# 1. Parse safety literature
lobster: Parse publications about drug adverse effects

# 2. Extract drugs and outcomes
lobster: Extract drugs, interactions, and outcomes

# 3. Focus on adverse relationships
lobster: Extract "causes", "interacts_with", and "contraindicated" relationships

# 4. Build adverse effect network
lobster: Build knowledge graph

# 5. Query for safety concerns
lobster: Find all adverse effects caused by drug X
lobster: Find all drug-drug interactions
```

## Integration with Other Lobster Agents

### With Research Agent

```python
# 1. Use research agent to find relevant papers
lobster: Find papers about aspirin and cardiovascular disease

# 2. Extract PMIDs and parse
lobster: Parse the top 5 papers

# 3. Build knowledge graph
lobster: Build combined knowledge graph from these papers
```

### With Data Expert

```python
# 1. Download gene expression data related to drug
lobster: Download GEO dataset GSE12345

# 2. Parse drug mechanism publications
lobster: Parse publications about the drug mechanism

# 3. Build knowledge graph of drug-gene relationships
lobster: Build knowledge graph connecting drug targets to gene expression
```

### With Visualization Expert

```python
# 1. Build knowledge graph
lobster: Build pharmacology knowledge graph

# 2. Create custom visualization
lobster: Create custom network visualization with advanced styling
```

## Best Practices

### 1. Publication Selection
- Use high-quality, peer-reviewed publications
- Focus on specific research questions
- Include review articles for comprehensive coverage
- Verify publication accessibility

### 2. Entity Extraction
- Start with default confidence threshold (0.5)
- Increase threshold (0.7-0.9) for high-precision needs
- Specify entity types for focused analysis
- Review extracted entities for accuracy

### 3. Relationship Extraction
- Focus on specific relationship types for targeted analysis
- Use context to validate relationships
- Check confidence scores for reliability
- Combine multiple publications for robust relationships

### 4. Graph Construction
- Use descriptive graph names for organization
- Build multiple graphs for comparison
- Validate graph statistics (nodes, edges, density)
- Check for disconnected components

### 5. Visualization
- Choose layout appropriate for graph structure:
  - Force: General purpose, shows clusters
  - Circular: Good for small to medium graphs
  - Hierarchical: Shows entity type organization
- Size nodes by importance (mentions, degree)
- Color by entity type for clarity

### 6. Export
- Choose format based on downstream use:
  - JSON: General purpose, web apps
  - RDF: Semantic web integration
  - Neo4j: Graph database storage
  - NetworkX: Python analysis
  - GraphML: External visualization tools

### 7. Query
- Start with simple node/edge queries
- Use filters to narrow results
- Explore paths for mechanism discovery
- Extract subgraphs for focused analysis

## Limitations and Considerations

### Current Limitations

1. **Entity Recognition**
   - Pattern-based extraction (simplified NLP)
   - May miss complex entity names
   - Limited context awareness
   - No disambiguation of ambiguous terms

2. **Relationship Extraction**
   - Co-occurrence based (simplified)
   - May miss negated relationships
   - Limited to sentence-level context
   - May produce false positives

3. **Graph Construction**
   - No automatic entity linking to databases
   - No confidence aggregation across mentions
   - Simple deduplication by text matching
   - No hierarchy or ontology integration

4. **Scalability**
   - Designed for moderate-sized publications
   - Memory-based graph construction
   - No distributed processing
   - Single publication focus

### Future Enhancements

Planned improvements:

- **Advanced NLP**: Integration with BioBERT, PubMedBERT, or similar models
- **Entity Linking**: Automatic linking to ChEMBL, DrugBank, UniProt
- **Ontology Integration**: MeSH, ChEBI, Gene Ontology integration
- **Machine Learning**: ML-based entity and relationship extraction
- **Database Integration**: Direct queries to pharmacology databases
- **Batch Processing**: Handle multiple publications efficiently
- **Graph Analytics**: Centrality, community detection, path analysis
- **Temporal Analysis**: Track knowledge evolution over time

## Troubleshooting

### Common Issues

**Issue**: No entities extracted
- **Solution**: Lower confidence threshold, check publication text quality

**Issue**: Few relationships found
- **Solution**: Ensure entities are extracted first, check relationship types

**Issue**: Graph visualization is cluttered
- **Solution**: Filter entities, use hierarchical layout, export for Gephi

**Issue**: Export format not compatible
- **Solution**: Verify format name, check output path permissions

**Issue**: Query returns no results
- **Solution**: Verify graph name, check filter syntax, use broader filters

### Error Messages

- **"Publication 'X' not found"**: Parse the publication first
- **"Entities for 'X' not found"**: Extract entities before relationships
- **"Knowledge graph 'X' not found"**: Build graph before visualization/export
- **"Unknown export format"**: Use valid format (json, rdf, neo4j, networkx, graphml)
- **"Unknown query type"**: Use valid type (node, edge, path, subgraph)

## Technical Details

### Architecture

```
PharmacologyKnowledgeGraphExpert (Agent)
├── PharmacologyKnowledgeGraphService (Stateless Service)
│   ├── parse_publication()
│   ├── extract_entities()
│   ├── extract_relationships()
│   ├── build_knowledge_graph()
│   ├── visualize_knowledge_graph()
│   ├── export_knowledge_graph()
│   └── query_knowledge_graph()
└── Tools (Agent Tools)
    ├── parse_pharmacology_publication
    ├── extract_pharmacology_entities
    ├── extract_pharmacology_relationships
    ├── build_pharmacology_knowledge_graph
    ├── visualize_pharmacology_knowledge_graph
    ├── export_pharmacology_knowledge_graph
    ├── query_pharmacology_knowledge_graph
    └── create_pharmacology_kg_summary
```

### Data Flow

1. **Input**: Publication text (PubMed, DOI, full text)
2. **Parsing**: Extract sections, sentences, metadata
3. **Entity Extraction**: Pattern matching, confidence scoring
4. **Relationship Extraction**: Co-occurrence analysis, dependency patterns
5. **Graph Construction**: Node and edge creation, statistics
6. **Visualization**: Layout calculation, interactive plots
7. **Export**: Format conversion, file writing
8. **Query**: Pattern matching, graph traversal

### Storage

All results are stored in the DataManagerV2 system:

- **Parsed Publications**: `metadata_store['parsed_publication_{id}']`
- **Entities**: `metadata_store['entities_{id}']`
- **Relationships**: `metadata_store['relationships_{id}']`
- **Knowledge Graphs**: `metadata_store['knowledge_graph_{name}']`
- **Provenance**: Complete operation history in tool usage log

## References

### Pharmacology Databases

- **ChEMBL**: Bioactive molecules database
- **DrugBank**: Drug and drug target database
- **UniProt**: Protein sequence and functional information
- **Reactome**: Pathway database
- **KEGG**: Pathway and disease database
- **STRING**: Protein-protein interaction networks
- **BioGRID**: Biological interaction database

### Ontologies

- **MeSH**: Medical Subject Headings
- **ChEBI**: Chemical Entities of Biological Interest
- **Gene Ontology**: Gene function classification
- **Disease Ontology**: Disease classification

### Related Tools

- **Neo4j**: Graph database for storage
- **Gephi**: Graph visualization software
- **Cytoscape**: Network visualization platform
- **NetworkX**: Python graph analysis library

## Examples

### Example 1: Aspirin Mechanism

```python
lobster: Parse this text: "Aspirin irreversibly inhibits cyclooxygenase-1 (COX-1) and cyclooxygenase-2 (COX-2) enzymes. COX-2 inhibition reduces inflammation and pain. This mechanism treats cardiovascular disease by preventing platelet aggregation."

lobster: Extract entities
# Entities found: aspirin (drug), COX-1 (target), COX-2 (target), inflammation (outcome), cardiovascular disease (disease)

lobster: Extract relationships
# Relationships: aspirin inhibits COX-1, aspirin inhibits COX-2, aspirin treats cardiovascular disease

lobster: Build knowledge graph named "aspirin_mechanism"

lobster: Visualize the knowledge graph

lobster: Export as JSON
```

### Example 2: Drug Interaction Network

```python
lobster: Parse publications about warfarin drug interactions

lobster: Extract drugs and interaction types

lobster: Extract "interacts_with" relationships

lobster: Build knowledge graph named "warfarin_interactions"

lobster: Query for all drugs that interact with warfarin

lobster: Export for Neo4j database
```

### Example 3: Target-Disease Network

```python
lobster: Parse publications about EGFR inhibitors in cancer

lobster: Extract drugs, targets, and diseases

lobster: Build knowledge graph

lobster: Find all EGFR-targeting drugs

lobster: Find all cancers treated by EGFR inhibitors

lobster: Export as GraphML for Cytoscape
```

## Conclusion

The Pharmacology Knowledge Graph agent provides a comprehensive system for:
- Extracting structured knowledge from pharmacology literature
- Building and analyzing drug-target-disease networks
- Supporting drug discovery and repurposing
- Enabling mechanism of action analysis
- Facilitating safety and interaction studies

Use this agent to transform unstructured pharmacology literature into structured, queryable knowledge graphs for research and analysis.

## See Also

- [Research Agent](research-agent.md): Literature discovery
- [Method Expert](method-expert.md): Computational method extraction
- [Visualization Expert](visualization-expert.md): Advanced visualizations
- [Data Expert](data-expert.md): Data management and integration
