"""
Pharmacology Knowledge Graph Service for parsing publications and constructing knowledge graphs.

This service implements professional-grade pharmacological entity extraction, relationship mining,
and knowledge graph construction from pharmacology publications using NLP and text mining.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import anndata
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class PharmacologyKnowledgeGraphError(Exception):
    """Base exception for pharmacology knowledge graph operations."""
    pass


class EntityExtractionError(PharmacologyKnowledgeGraphError):
    """Raised when entity extraction fails."""
    pass


class GraphConstructionError(PharmacologyKnowledgeGraphError):
    """Raised when graph construction fails."""
    pass


# Entity types for pharmacology knowledge graphs
ENTITY_TYPES = {
    'drug': ['drug', 'compound', 'molecule', 'medication', 'pharmaceutical'],
    'target': ['protein', 'gene', 'receptor', 'enzyme', 'kinase', 'channel'],
    'disease': ['disease', 'disorder', 'syndrome', 'condition', 'pathology'],
    'mechanism': ['inhibition', 'activation', 'modulation', 'agonism', 'antagonism'],
    'interaction': ['interaction', 'synergy', 'antagonism', 'potentiation'],
    'outcome': ['efficacy', 'toxicity', 'adverse', 'response', 'effect'],
}

# Relationship types
RELATIONSHIP_TYPES = [
    'inhibits', 'activates', 'binds_to', 'treats', 'indicated_for',
    'contraindicated', 'interacts_with', 'metabolizes', 'targets',
    'associated_with', 'causes', 'prevents'
]


class PharmacologyKnowledgeGraphService:
    """
    Advanced knowledge graph construction service for pharmacology publications.

    This stateless service provides methods for parsing publications, extracting
    pharmacological entities, identifying relationships, and constructing knowledge
    graphs following best practices from biomedical NLP and knowledge representation.
    """

    def __init__(self):
        """
        Initialize the pharmacology knowledge graph service.

        This service is stateless and doesn't require a data manager instance.
        """
        logger.debug("Initializing stateless PharmacologyKnowledgeGraphService")
        logger.debug("PharmacologyKnowledgeGraphService initialized successfully")

    def parse_publication(
        self,
        publication_text: str,
        publication_id: Optional[str] = None,
        source_type: str = "full_text"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Parse a pharmacology publication and extract structured content.

        Args:
            publication_text: Full text or abstract of the publication
            publication_id: Optional publication identifier (PMID, DOI)
            source_type: Type of source ("full_text", "abstract", "pdf")

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Parsed content and parsing stats

        Raises:
            PharmacologyKnowledgeGraphError: If parsing fails
        """
        try:
            logger.info(f"Parsing publication: {publication_id or 'unknown'}")

            # Extract sections from publication text
            sections = self._extract_sections(publication_text)

            # Extract metadata
            metadata = {
                'publication_id': publication_id,
                'source_type': source_type,
                'parsed_at': datetime.now().isoformat(),
                'text_length': len(publication_text),
                'sections_found': list(sections.keys())
            }

            # Extract sentences for entity recognition
            sentences = self._extract_sentences(publication_text)

            parsed_content = {
                'metadata': metadata,
                'sections': sections,
                'sentences': sentences,
                'raw_text': publication_text
            }

            parsing_stats = {
                'publication_id': publication_id,
                'source_type': source_type,
                'sections_extracted': len(sections),
                'sentences_extracted': len(sentences),
                'text_length': len(publication_text),
                'analysis_type': 'publication_parsing'
            }

            logger.info(f"Publication parsed successfully: {len(sentences)} sentences")
            return parsed_content, parsing_stats

        except Exception as e:
            logger.exception(f"Error parsing publication: {e}")
            raise PharmacologyKnowledgeGraphError(f"Publication parsing failed: {str(e)}")

    def extract_entities(
        self,
        parsed_content: Dict[str, Any],
        entity_types: Optional[List[str]] = None,
        confidence_threshold: float = 0.5
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
        """
        Extract pharmacological entities from parsed publication content.

        Args:
            parsed_content: Parsed publication content from parse_publication()
            entity_types: List of entity types to extract (None = all types)
            confidence_threshold: Minimum confidence score for entity extraction

        Returns:
            Tuple[Dict[str, List[Dict]], Dict]: Extracted entities and extraction stats

        Raises:
            EntityExtractionError: If entity extraction fails
        """
        try:
            logger.info("Extracting pharmacological entities")

            entity_types = entity_types or list(ENTITY_TYPES.keys())
            sentences = parsed_content.get('sentences', [])

            # Extract entities using pattern matching and dictionary lookup
            entities_by_type = defaultdict(list)

            for entity_type in entity_types:
                entities = self._extract_entities_by_type(
                    sentences,
                    entity_type,
                    confidence_threshold
                )
                entities_by_type[entity_type].extend(entities)

            # Deduplicate and enrich entities
            entities_by_type = self._deduplicate_entities(entities_by_type)

            extraction_stats = {
                'entity_types_extracted': list(entities_by_type.keys()),
                'total_entities': sum(len(ents) for ents in entities_by_type.values()),
                'entities_by_type': {
                    etype: len(ents) for etype, ents in entities_by_type.items()
                },
                'confidence_threshold': confidence_threshold,
                'analysis_type': 'entity_extraction'
            }

            logger.info(f"Extracted {extraction_stats['total_entities']} entities")
            return dict(entities_by_type), extraction_stats

        except Exception as e:
            logger.exception(f"Error extracting entities: {e}")
            raise EntityExtractionError(f"Entity extraction failed: {str(e)}")

    def extract_relationships(
        self,
        parsed_content: Dict[str, Any],
        entities: Dict[str, List[Dict[str, Any]]],
        relationship_types: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extract relationships between pharmacological entities.

        Args:
            parsed_content: Parsed publication content
            entities: Extracted entities from extract_entities()
            relationship_types: List of relationship types to extract

        Returns:
            Tuple[List[Dict], Dict]: Extracted relationships and extraction stats

        Raises:
            EntityExtractionError: If relationship extraction fails
        """
        try:
            logger.info("Extracting entity relationships")

            relationship_types = relationship_types or RELATIONSHIP_TYPES
            sentences = parsed_content.get('sentences', [])

            # Extract relationships using dependency parsing patterns
            relationships = self._extract_relationships_from_sentences(
                sentences,
                entities,
                relationship_types
            )

            # Filter and validate relationships
            relationships = self._validate_relationships(relationships)

            extraction_stats = {
                'total_relationships': len(relationships),
                'relationships_by_type': self._count_by_type(relationships, 'relation_type'),
                'unique_entity_pairs': len(set((r['source'], r['target']) for r in relationships)),
                'analysis_type': 'relationship_extraction'
            }

            logger.info(f"Extracted {len(relationships)} relationships")
            return relationships, extraction_stats

        except Exception as e:
            logger.exception(f"Error extracting relationships: {e}")
            raise EntityExtractionError(f"Relationship extraction failed: {str(e)}")

    def build_knowledge_graph(
        self,
        entities: Dict[str, List[Dict[str, Any]]],
        relationships: List[Dict[str, Any]],
        graph_name: str = "pharmacology_kg"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Construct a knowledge graph from extracted entities and relationships.

        Args:
            entities: Extracted entities by type
            relationships: Extracted relationships
            graph_name: Name for the knowledge graph

        Returns:
            Tuple[Dict, Dict]: Knowledge graph structure and construction stats

        Raises:
            GraphConstructionError: If graph construction fails
        """
        try:
            logger.info(f"Building knowledge graph: {graph_name}")

            # Create node list
            nodes = []
            node_index = {}
            node_id = 0

            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    node_data = {
                        'id': node_id,
                        'name': entity['text'],
                        'type': entity_type,
                        'properties': entity.get('properties', {}),
                        'mentions': entity.get('mentions', 1)
                    }
                    nodes.append(node_data)
                    node_index[entity['text']] = node_id
                    node_id += 1

            # Create edge list
            edges = []
            for rel in relationships:
                source_id = node_index.get(rel['source'])
                target_id = node_index.get(rel['target'])

                if source_id is not None and target_id is not None:
                    edge_data = {
                        'source': source_id,
                        'target': target_id,
                        'relation_type': rel['relation_type'],
                        'confidence': rel.get('confidence', 1.0),
                        'evidence': rel.get('evidence', '')
                    }
                    edges.append(edge_data)

            # Calculate graph statistics
            graph_stats = self._calculate_graph_statistics(nodes, edges)

            knowledge_graph = {
                'name': graph_name,
                'nodes': nodes,
                'edges': edges,
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'n_nodes': len(nodes),
                    'n_edges': len(edges),
                    'node_types': list(entities.keys())
                }
            }

            construction_stats = {
                'graph_name': graph_name,
                'n_nodes': len(nodes),
                'n_edges': len(edges),
                'nodes_by_type': {
                    etype: len(elist) for etype, elist in entities.items()
                },
                'edges_by_type': self._count_by_type(edges, 'relation_type'),
                **graph_stats,
                'analysis_type': 'knowledge_graph_construction'
            }

            logger.info(f"Knowledge graph built: {len(nodes)} nodes, {len(edges)} edges")
            return knowledge_graph, construction_stats

        except Exception as e:
            logger.exception(f"Error building knowledge graph: {e}")
            raise GraphConstructionError(f"Knowledge graph construction failed: {str(e)}")

    def visualize_knowledge_graph(
        self,
        knowledge_graph: Dict[str, Any],
        layout: str = "force",
        node_size_by: str = "mentions",
        color_by: str = "type"
    ) -> Tuple[go.Figure, Dict[str, Any]]:
        """
        Create an interactive visualization of the knowledge graph.

        Args:
            knowledge_graph: Knowledge graph from build_knowledge_graph()
            layout: Graph layout algorithm ("force", "circular", "hierarchical")
            node_size_by: Node attribute to determine size
            color_by: Node attribute for coloring

        Returns:
            Tuple[go.Figure, Dict]: Plotly figure and visualization stats

        Raises:
            GraphConstructionError: If visualization fails
        """
        try:
            logger.info("Creating knowledge graph visualization")

            nodes = knowledge_graph['nodes']
            edges = knowledge_graph['edges']

            # Calculate layout positions
            node_positions = self._calculate_layout_positions(nodes, edges, layout)

            # Create edge traces
            edge_traces = self._create_edge_traces(edges, node_positions)

            # Create node trace
            node_trace = self._create_node_trace(
                nodes,
                node_positions,
                node_size_by,
                color_by
            )

            # Create figure
            fig = go.Figure(data=[edge_traces, node_trace])

            fig.update_layout(
                title=f"Pharmacology Knowledge Graph: {knowledge_graph['name']}",
                showlegend=True,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                height=800
            )

            viz_stats = {
                'layout_type': layout,
                'n_nodes_displayed': len(nodes),
                'n_edges_displayed': len(edges),
                'node_size_by': node_size_by,
                'color_by': color_by,
                'analysis_type': 'knowledge_graph_visualization'
            }

            logger.info("Knowledge graph visualization created")
            return fig, viz_stats

        except Exception as e:
            logger.exception(f"Error creating visualization: {e}")
            raise GraphConstructionError(f"Visualization failed: {str(e)}")

    def export_knowledge_graph(
        self,
        knowledge_graph: Dict[str, Any],
        export_format: str = "json",
        output_path: Optional[Path] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Export knowledge graph in various formats.

        Args:
            knowledge_graph: Knowledge graph to export
            export_format: Export format ("json", "rdf", "neo4j", "networkx", "graphml")
            output_path: Optional path for export file

        Returns:
            Tuple[str, Dict]: Export result (path or serialized data) and export stats

        Raises:
            GraphConstructionError: If export fails
        """
        try:
            logger.info(f"Exporting knowledge graph as {export_format}")

            if export_format == "json":
                result = self._export_json(knowledge_graph, output_path)
            elif export_format == "rdf":
                result = self._export_rdf(knowledge_graph, output_path)
            elif export_format == "neo4j":
                result = self._export_neo4j(knowledge_graph, output_path)
            elif export_format == "networkx":
                result = self._export_networkx(knowledge_graph, output_path)
            elif export_format == "graphml":
                result = self._export_graphml(knowledge_graph, output_path)
            else:
                raise GraphConstructionError(f"Unknown export format: {export_format}")

            export_stats = {
                'export_format': export_format,
                'output_path': str(result) if output_path else 'in_memory',
                'n_nodes': len(knowledge_graph['nodes']),
                'n_edges': len(knowledge_graph['edges']),
                'analysis_type': 'knowledge_graph_export'
            }

            logger.info(f"Knowledge graph exported as {export_format}")
            return result, export_stats

        except Exception as e:
            logger.exception(f"Error exporting knowledge graph: {e}")
            raise GraphConstructionError(f"Export failed: {str(e)}")

    def query_knowledge_graph(
        self,
        knowledge_graph: Dict[str, Any],
        query_type: str = "node",
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Query knowledge graph for specific nodes, edges, or patterns.

        Args:
            knowledge_graph: Knowledge graph to query
            query_type: Type of query ("node", "edge", "path", "subgraph")
            filters: Query filters (e.g., {'type': 'drug', 'name': 'aspirin'})

        Returns:
            Tuple[List[Dict], Dict]: Query results and query stats

        Raises:
            GraphConstructionError: If query fails
        """
        try:
            logger.info(f"Querying knowledge graph: {query_type}")

            filters = filters or {}

            if query_type == "node":
                results = self._query_nodes(knowledge_graph, filters)
            elif query_type == "edge":
                results = self._query_edges(knowledge_graph, filters)
            elif query_type == "path":
                results = self._query_paths(knowledge_graph, filters)
            elif query_type == "subgraph":
                results = self._query_subgraph(knowledge_graph, filters)
            else:
                raise GraphConstructionError(f"Unknown query type: {query_type}")

            query_stats = {
                'query_type': query_type,
                'filters': filters,
                'n_results': len(results),
                'analysis_type': 'knowledge_graph_query'
            }

            logger.info(f"Query returned {len(results)} results")
            return results, query_stats

        except Exception as e:
            logger.exception(f"Error querying knowledge graph: {e}")
            raise GraphConstructionError(f"Query failed: {str(e)}")

    # Helper methods
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from publication text."""
        sections = {}

        # Common section headers in pharmacology papers
        section_patterns = {
            'abstract': r'(?i)abstract[:\s]+(.*?)(?=\n\n|\nintroduction|\nmethods)',
            'introduction': r'(?i)introduction[:\s]+(.*?)(?=\n\nmethods|\n\nresults)',
            'methods': r'(?i)methods[:\s]+(.*?)(?=\n\nresults|\n\ndiscussion)',
            'results': r'(?i)results[:\s]+(.*?)(?=\n\ndiscussion|\n\nconclusion)',
            'discussion': r'(?i)discussion[:\s]+(.*?)(?=\n\nconclusion|\n\nreferences)',
            'conclusion': r'(?i)conclusion[:\s]+(.*?)(?=\n\nreferences|$)'
        }

        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
            if match:
                sections[section_name] = match.group(1).strip()

        # If no sections found, treat as single text
        if not sections:
            sections['full_text'] = text

        return sections

    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        # Simple sentence splitting (can be enhanced with NLTK/spaCy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _extract_entities_by_type(
        self,
        sentences: List[str],
        entity_type: str,
        confidence_threshold: float
    ) -> List[Dict[str, Any]]:
        """Extract entities of a specific type from sentences."""
        entities = []
        entity_patterns = ENTITY_TYPES.get(entity_type, [])

        for sentence in sentences:
            # Pattern-based extraction (simplified)
            for pattern in entity_patterns:
                matches = re.finditer(
                    r'\b' + pattern + r'\w*\b',
                    sentence,
                    re.IGNORECASE
                )

                for match in matches:
                    entity_text = match.group(0)
                    entities.append({
                        'text': entity_text,
                        'type': entity_type,
                        'confidence': 0.7,  # Simplified confidence
                        'context': sentence,
                        'mentions': 1
                    })

        return entities

    def _deduplicate_entities(
        self,
        entities_by_type: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Deduplicate and merge entities."""
        deduplicated = {}

        for entity_type, entities in entities_by_type.items():
            seen = {}
            for entity in entities:
                text_lower = entity['text'].lower()
                if text_lower in seen:
                    seen[text_lower]['mentions'] += 1
                else:
                    seen[text_lower] = entity

            deduplicated[entity_type] = list(seen.values())

        return deduplicated

    def _extract_relationships_from_sentences(
        self,
        sentences: List[str],
        entities: Dict[str, List[Dict[str, Any]]],
        relationship_types: List[str]
    ) -> List[Dict[str, Any]]:
        """Extract relationships from sentences using pattern matching."""
        relationships = []

        # Create entity lookup
        entity_texts = set()
        for entity_list in entities.values():
            for entity in entity_list:
                entity_texts.add(entity['text'].lower())

        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Find entity co-occurrences
            found_entities = [e for e in entity_texts if e in sentence_lower]

            if len(found_entities) >= 2:
                # Extract relationships between co-occurring entities
                for rel_type in relationship_types:
                    if rel_type in sentence_lower:
                        # Create relationship for first two entities (simplified)
                        relationships.append({
                            'source': found_entities[0],
                            'target': found_entities[1],
                            'relation_type': rel_type,
                            'confidence': 0.6,
                            'evidence': sentence
                        })

        return relationships

    def _validate_relationships(
        self,
        relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate and filter relationships."""
        # Remove duplicates and low-confidence relationships
        validated = []
        seen = set()

        for rel in relationships:
            key = (rel['source'], rel['target'], rel['relation_type'])
            if key not in seen and rel.get('confidence', 0) > 0.5:
                validated.append(rel)
                seen.add(key)

        return validated

    def _calculate_graph_statistics(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate basic graph statistics."""
        # Calculate degree centrality
        degree = defaultdict(int)
        for edge in edges:
            degree[edge['source']] += 1
            degree[edge['target']] += 1

        avg_degree = np.mean(list(degree.values())) if degree else 0
        max_degree = max(degree.values()) if degree else 0

        return {
            'avg_degree': float(avg_degree),
            'max_degree': int(max_degree),
            'density': len(edges) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0
        }

    def _count_by_type(
        self,
        items: List[Dict[str, Any]],
        type_key: str
    ) -> Dict[str, int]:
        """Count items by type."""
        counts = defaultdict(int)
        for item in items:
            counts[item.get(type_key, 'unknown')] += 1
        return dict(counts)

    def _calculate_layout_positions(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        layout: str
    ) -> Dict[int, Tuple[float, float]]:
        """Calculate node positions for visualization."""
        positions = {}

        if layout == "circular":
            # Circular layout
            n = len(nodes)
            for i, node in enumerate(nodes):
                angle = 2 * np.pi * i / n
                positions[node['id']] = (np.cos(angle), np.sin(angle))

        elif layout == "force":
            # Simplified force-directed layout
            # In production, use networkx spring_layout
            for i, node in enumerate(nodes):
                positions[node['id']] = (
                    np.random.randn() * 0.5,
                    np.random.randn() * 0.5
                )

        else:  # hierarchical
            # Simple hierarchical layout by type
            type_y = {}
            type_counts = defaultdict(int)

            for i, node in enumerate(nodes):
                node_type = node['type']
                if node_type not in type_y:
                    type_y[node_type] = len(type_y)

                x = type_counts[node_type]
                y = type_y[node_type]
                positions[node['id']] = (x, y)
                type_counts[node_type] += 1

        return positions

    def _create_edge_traces(
        self,
        edges: List[Dict[str, Any]],
        positions: Dict[int, Tuple[float, float]]
    ) -> go.Scatter:
        """Create edge traces for visualization."""
        edge_x = []
        edge_y = []

        for edge in edges:
            x0, y0 = positions[edge['source']]
            x1, y1 = positions[edge['target']]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        return go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

    def _create_node_trace(
        self,
        nodes: List[Dict[str, Any]],
        positions: Dict[int, Tuple[float, float]],
        node_size_by: str,
        color_by: str
    ) -> go.Scatter:
        """Create node trace for visualization."""
        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        node_colors = []

        for node in nodes:
            x, y = positions[node['id']]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node['name']}<br>Type: {node['type']}")

            # Size by attribute
            size = node.get(node_size_by, 10)
            node_sizes.append(min(size * 5, 50))  # Scale and cap size

            # Color by type (simplified)
            type_colors = {
                'drug': 'red', 'target': 'blue', 'disease': 'green',
                'mechanism': 'purple', 'interaction': 'orange', 'outcome': 'brown'
            }
            node_colors.append(type_colors.get(node['type'], 'gray'))

        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[n['name'] for n in nodes],
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line_width=2
            )
        )

    def _export_json(
        self,
        knowledge_graph: Dict[str, Any],
        output_path: Optional[Path]
    ) -> str:
        """Export knowledge graph as JSON."""
        json_str = json.dumps(knowledge_graph, indent=2)

        if output_path:
            output_path.write_text(json_str)
            return str(output_path)

        return json_str

    def _export_rdf(
        self,
        knowledge_graph: Dict[str, Any],
        output_path: Optional[Path]
    ) -> str:
        """Export knowledge graph as RDF/Turtle."""
        # Simplified RDF export
        rdf_lines = ["@prefix kg: <http://pharmacology.kg/> ."]

        for node in knowledge_graph['nodes']:
            rdf_lines.append(
                f"kg:node{node['id']} a kg:{node['type']} ; "
                f"kg:name \"{node['name']}\" ."
            )

        for edge in knowledge_graph['edges']:
            rdf_lines.append(
                f"kg:node{edge['source']} kg:{edge['relation_type']} kg:node{edge['target']} ."
            )

        rdf_str = '\n'.join(rdf_lines)

        if output_path:
            output_path.write_text(rdf_str)
            return str(output_path)

        return rdf_str

    def _export_neo4j(
        self,
        knowledge_graph: Dict[str, Any],
        output_path: Optional[Path]
    ) -> str:
        """Export knowledge graph as Neo4j Cypher queries."""
        cypher_lines = []

        # Create nodes
        for node in knowledge_graph['nodes']:
            cypher_lines.append(
                f"CREATE (n{node['id']}:{node['type']} "
                f"{{name: '{node['name']}', mentions: {node['mentions']}}})"
            )

        # Create relationships
        for edge in knowledge_graph['edges']:
            cypher_lines.append(
                f"CREATE (n{edge['source']})-[:{edge['relation_type'].upper()} "
                f"{{confidence: {edge['confidence']}}}]->(n{edge['target']})"
            )

        cypher_str = ';\n'.join(cypher_lines) + ';'

        if output_path:
            output_path.write_text(cypher_str)
            return str(output_path)

        return cypher_str

    def _export_networkx(
        self,
        knowledge_graph: Dict[str, Any],
        output_path: Optional[Path]
    ) -> str:
        """Export knowledge graph as NetworkX compatible format."""
        # Export as edge list format compatible with NetworkX
        edge_list = []

        for edge in knowledge_graph['edges']:
            source_node = knowledge_graph['nodes'][edge['source']]
            target_node = knowledge_graph['nodes'][edge['target']]
            edge_list.append(
                f"{source_node['name']}\t{target_node['name']}\t{edge['relation_type']}"
            )

        edge_list_str = '\n'.join(edge_list)

        if output_path:
            output_path.write_text(edge_list_str)
            return str(output_path)

        return edge_list_str

    def _export_graphml(
        self,
        knowledge_graph: Dict[str, Any],
        output_path: Optional[Path]
    ) -> str:
        """Export knowledge graph as GraphML."""
        # Simplified GraphML export
        graphml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
            '  <graph id="G" edgedefault="directed">'
        ]

        for node in knowledge_graph['nodes']:
            graphml_lines.append(
                f'    <node id="{node["id"]}">'
                f'<data key="name">{node["name"]}</data>'
                f'<data key="type">{node["type"]}</data>'
                f'</node>'
            )

        for edge in knowledge_graph['edges']:
            graphml_lines.append(
                f'    <edge source="{edge["source"]}" target="{edge["target"]}">'
                f'<data key="type">{edge["relation_type"]}</data>'
                f'</edge>'
            )

        graphml_lines.extend(['  </graph>', '</graphml>'])
        graphml_str = '\n'.join(graphml_lines)

        if output_path:
            output_path.write_text(graphml_str)
            return str(output_path)

        return graphml_str

    def _query_nodes(
        self,
        knowledge_graph: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Query nodes with filters."""
        results = []

        for node in knowledge_graph['nodes']:
            match = True
            for key, value in filters.items():
                if node.get(key) != value:
                    match = False
                    break

            if match:
                results.append(node)

        return results

    def _query_edges(
        self,
        knowledge_graph: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Query edges with filters."""
        results = []

        for edge in knowledge_graph['edges']:
            match = True
            for key, value in filters.items():
                if edge.get(key) != value:
                    match = False
                    break

            if match:
                results.append(edge)

        return results

    def _query_paths(
        self,
        knowledge_graph: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Query paths between nodes."""
        # Simplified path finding (BFS)
        source = filters.get('source')
        target = filters.get('target')
        max_length = filters.get('max_length', 3)

        if not source or not target:
            return []

        # Build adjacency list
        adj = defaultdict(list)
        for edge in knowledge_graph['edges']:
            adj[edge['source']].append((edge['target'], edge))

        # BFS to find paths
        paths = []
        queue = [(source, [source], [])]

        while queue:
            current, path, edges = queue.pop(0)

            if len(path) > max_length:
                continue

            if current == target:
                paths.append({'nodes': path, 'edges': edges})
                continue

            for neighbor, edge in adj[current]:
                if neighbor not in path:
                    queue.append((neighbor, path + [neighbor], edges + [edge]))

        return paths

    def _query_subgraph(
        self,
        knowledge_graph: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract subgraph based on filters."""
        # Get matching nodes
        matching_nodes = self._query_nodes(knowledge_graph, filters)
        matching_node_ids = {n['id'] for n in matching_nodes}

        # Get edges between matching nodes
        matching_edges = [
            e for e in knowledge_graph['edges']
            if e['source'] in matching_node_ids and e['target'] in matching_node_ids
        ]

        return [{
            'nodes': matching_nodes,
            'edges': matching_edges
        }]
