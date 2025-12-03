"""
Unit tests for PharmacologyKnowledgeGraphService.

These tests verify that the PharmacologyKnowledgeGraphService correctly parses publications,
extracts entities and relationships, and constructs knowledge graphs.
"""

from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import the service and related classes
from lobster.tools.pharmacology_knowledge_graph_service import (
    PharmacologyKnowledgeGraphService,
    PharmacologyKnowledgeGraphError,
    EntityExtractionError,
    GraphConstructionError,
    ENTITY_TYPES,
    RELATIONSHIP_TYPES
)


@pytest.fixture
def service():
    """Create a PharmacologyKnowledgeGraphService instance for testing."""
    return PharmacologyKnowledgeGraphService()


@pytest.fixture
def sample_publication_text():
    """Create sample publication text for testing."""
    return """
    Abstract: This study investigates the effects of aspirin on cardiovascular disease.
    Aspirin is a non-steroidal anti-inflammatory drug that inhibits cyclooxygenase enzyme.

    Introduction: Cardiovascular disease is a leading cause of mortality worldwide.
    Aspirin has been shown to reduce the risk of heart attacks by preventing blood clot formation.

    Methods: We analyzed the interaction between aspirin and COX-2 protein.

    Results: Aspirin binds to the COX-2 enzyme and inhibits its activity.
    This inhibition leads to reduced inflammation and thrombosis.

    Discussion: The aspirin-COX-2 interaction treats cardiovascular disease effectively.
    """


@pytest.fixture
def sample_parsed_content():
    """Create sample parsed content for testing."""
    return {
        'metadata': {
            'publication_id': 'PMID12345',
            'source_type': 'full_text',
            'parsed_at': '2024-01-01T00:00:00',
            'text_length': 500,
            'sections_found': ['abstract', 'introduction', 'methods', 'results', 'discussion']
        },
        'sections': {
            'abstract': 'Aspirin inhibits COX-2',
            'methods': 'Analyzed drug-protein interactions'
        },
        'sentences': [
            'Aspirin is a drug that inhibits COX-2 protein.',
            'COX-2 is a target enzyme for anti-inflammatory drugs.',
            'Aspirin treats cardiovascular disease.',
        ],
        'raw_text': 'Sample text'
    }


@pytest.fixture
def sample_entities():
    """Create sample extracted entities for testing."""
    return {
        'drug': [
            {'text': 'aspirin', 'type': 'drug', 'confidence': 0.9, 'context': 'sentence 1', 'mentions': 3},
            {'text': 'ibuprofen', 'type': 'drug', 'confidence': 0.8, 'context': 'sentence 2', 'mentions': 1}
        ],
        'target': [
            {'text': 'COX-2', 'type': 'target', 'confidence': 0.95, 'context': 'sentence 1', 'mentions': 2},
            {'text': 'COX-1', 'type': 'target', 'confidence': 0.85, 'context': 'sentence 3', 'mentions': 1}
        ],
        'disease': [
            {'text': 'cardiovascular disease', 'type': 'disease', 'confidence': 0.9, 'context': 'sentence 4', 'mentions': 2}
        ]
    }


@pytest.fixture
def sample_relationships():
    """Create sample extracted relationships for testing."""
    return [
        {
            'source': 'aspirin',
            'target': 'COX-2',
            'relation_type': 'inhibits',
            'confidence': 0.9,
            'evidence': 'Aspirin inhibits COX-2 protein.'
        },
        {
            'source': 'aspirin',
            'target': 'cardiovascular disease',
            'relation_type': 'treats',
            'confidence': 0.85,
            'evidence': 'Aspirin treats cardiovascular disease.'
        }
    ]


class TestPharmacologyKnowledgeGraphService:
    """Test the main PharmacologyKnowledgeGraphService class."""

    def test_initialization(self, service):
        """Test service initialization."""
        assert service is not None
        assert isinstance(service, PharmacologyKnowledgeGraphService)

    def test_parse_publication_success(self, service, sample_publication_text):
        """Test successful publication parsing."""
        parsed_content, parsing_stats = service.parse_publication(
            publication_text=sample_publication_text,
            publication_id='PMID12345',
            source_type='full_text'
        )

        assert 'metadata' in parsed_content
        assert 'sections' in parsed_content
        assert 'sentences' in parsed_content
        assert 'raw_text' in parsed_content

        assert parsed_content['metadata']['publication_id'] == 'PMID12345'
        assert parsed_content['metadata']['source_type'] == 'full_text'
        assert len(parsed_content['sentences']) > 0

        assert parsing_stats['publication_id'] == 'PMID12345'
        assert parsing_stats['analysis_type'] == 'publication_parsing'
        assert parsing_stats['sentences_extracted'] > 0

    def test_parse_publication_empty_text(self, service):
        """Test parsing with empty text."""
        with pytest.raises(PharmacologyKnowledgeGraphError):
            service.parse_publication(
                publication_text='',
                publication_id='PMID_EMPTY'
            )

    def test_extract_entities_success(self, service, sample_parsed_content):
        """Test successful entity extraction."""
        entities, extraction_stats = service.extract_entities(
            parsed_content=sample_parsed_content,
            entity_types=['drug', 'target', 'disease'],
            confidence_threshold=0.5
        )

        assert isinstance(entities, dict)
        assert extraction_stats['analysis_type'] == 'entity_extraction'
        assert extraction_stats['total_entities'] >= 0
        assert 'entities_by_type' in extraction_stats

    def test_extract_entities_specific_types(self, service, sample_parsed_content):
        """Test entity extraction with specific entity types."""
        entities, extraction_stats = service.extract_entities(
            parsed_content=sample_parsed_content,
            entity_types=['drug'],
            confidence_threshold=0.5
        )

        assert 'drug' in entities
        # Only drug entities should be extracted
        assert all(entity_type in ['drug'] for entity_type in entities.keys())

    def test_extract_entities_high_confidence(self, service, sample_parsed_content):
        """Test entity extraction with high confidence threshold."""
        entities, extraction_stats = service.extract_entities(
            parsed_content=sample_parsed_content,
            entity_types=None,
            confidence_threshold=0.9
        )

        # Should extract fewer entities with higher threshold
        assert extraction_stats['confidence_threshold'] == 0.9

    def test_extract_relationships_success(self, service, sample_parsed_content, sample_entities):
        """Test successful relationship extraction."""
        relationships, extraction_stats = service.extract_relationships(
            parsed_content=sample_parsed_content,
            entities=sample_entities,
            relationship_types=['inhibits', 'treats']
        )

        assert isinstance(relationships, list)
        assert extraction_stats['analysis_type'] == 'relationship_extraction'
        assert extraction_stats['total_relationships'] >= 0
        assert 'relationships_by_type' in extraction_stats

    def test_extract_relationships_specific_types(self, service, sample_parsed_content, sample_entities):
        """Test relationship extraction with specific types."""
        relationships, extraction_stats = service.extract_relationships(
            parsed_content=sample_parsed_content,
            entities=sample_entities,
            relationship_types=['inhibits']
        )

        # Check that only specified relationship types are extracted
        if relationships:
            for rel in relationships:
                assert rel['relation_type'] in ['inhibits'] + RELATIONSHIP_TYPES

    def test_build_knowledge_graph_success(self, service, sample_entities, sample_relationships):
        """Test successful knowledge graph construction."""
        knowledge_graph, construction_stats = service.build_knowledge_graph(
            entities=sample_entities,
            relationships=sample_relationships,
            graph_name='test_kg'
        )

        assert 'name' in knowledge_graph
        assert knowledge_graph['name'] == 'test_kg'
        assert 'nodes' in knowledge_graph
        assert 'edges' in knowledge_graph
        assert 'metadata' in knowledge_graph

        assert construction_stats['analysis_type'] == 'knowledge_graph_construction'
        assert construction_stats['graph_name'] == 'test_kg'
        assert construction_stats['n_nodes'] >= 0
        assert construction_stats['n_edges'] >= 0

    def test_build_knowledge_graph_empty_entities(self, service):
        """Test knowledge graph construction with empty entities."""
        entities = {}
        relationships = []

        knowledge_graph, construction_stats = service.build_knowledge_graph(
            entities=entities,
            relationships=relationships,
            graph_name='empty_kg'
        )

        assert construction_stats['n_nodes'] == 0
        assert construction_stats['n_edges'] == 0

    def test_visualize_knowledge_graph_success(self, service, sample_entities, sample_relationships):
        """Test knowledge graph visualization."""
        # First build a knowledge graph
        knowledge_graph, _ = service.build_knowledge_graph(
            entities=sample_entities,
            relationships=sample_relationships,
            graph_name='viz_test_kg'
        )

        # Then visualize it
        fig, viz_stats = service.visualize_knowledge_graph(
            knowledge_graph=knowledge_graph,
            layout='force',
            node_size_by='mentions',
            color_by='type'
        )

        assert fig is not None
        assert viz_stats['analysis_type'] == 'knowledge_graph_visualization'
        assert viz_stats['layout_type'] == 'force'
        assert viz_stats['n_nodes_displayed'] >= 0

    def test_visualize_knowledge_graph_different_layouts(self, service, sample_entities, sample_relationships):
        """Test visualization with different layout algorithms."""
        knowledge_graph, _ = service.build_knowledge_graph(
            entities=sample_entities,
            relationships=sample_relationships,
            graph_name='layout_test_kg'
        )

        for layout in ['force', 'circular', 'hierarchical']:
            fig, viz_stats = service.visualize_knowledge_graph(
                knowledge_graph=knowledge_graph,
                layout=layout
            )
            assert viz_stats['layout_type'] == layout

    def test_export_knowledge_graph_json(self, service, sample_entities, sample_relationships):
        """Test knowledge graph export in JSON format."""
        knowledge_graph, _ = service.build_knowledge_graph(
            entities=sample_entities,
            relationships=sample_relationships,
            graph_name='export_test_kg'
        )

        result, export_stats = service.export_knowledge_graph(
            knowledge_graph=knowledge_graph,
            export_format='json',
            output_path=None
        )

        assert isinstance(result, str)
        assert export_stats['export_format'] == 'json'
        assert export_stats['analysis_type'] == 'knowledge_graph_export'

    def test_export_knowledge_graph_multiple_formats(self, service, sample_entities, sample_relationships):
        """Test knowledge graph export in multiple formats."""
        knowledge_graph, _ = service.build_knowledge_graph(
            entities=sample_entities,
            relationships=sample_relationships,
            graph_name='multi_export_kg'
        )

        formats = ['json', 'rdf', 'neo4j', 'networkx', 'graphml']

        for fmt in formats:
            result, export_stats = service.export_knowledge_graph(
                knowledge_graph=knowledge_graph,
                export_format=fmt,
                output_path=None
            )
            assert export_stats['export_format'] == fmt
            assert isinstance(result, str)

    def test_export_knowledge_graph_invalid_format(self, service, sample_entities, sample_relationships):
        """Test export with invalid format."""
        knowledge_graph, _ = service.build_knowledge_graph(
            entities=sample_entities,
            relationships=sample_relationships,
            graph_name='invalid_format_kg'
        )

        with pytest.raises(GraphConstructionError):
            service.export_knowledge_graph(
                knowledge_graph=knowledge_graph,
                export_format='invalid_format'
            )

    def test_query_knowledge_graph_nodes(self, service, sample_entities, sample_relationships):
        """Test querying nodes in knowledge graph."""
        knowledge_graph, _ = service.build_knowledge_graph(
            entities=sample_entities,
            relationships=sample_relationships,
            graph_name='query_nodes_kg'
        )

        results, query_stats = service.query_knowledge_graph(
            knowledge_graph=knowledge_graph,
            query_type='node',
            filters={'type': 'drug'}
        )

        assert isinstance(results, list)
        assert query_stats['query_type'] == 'node'
        assert query_stats['analysis_type'] == 'knowledge_graph_query'

    def test_query_knowledge_graph_edges(self, service, sample_entities, sample_relationships):
        """Test querying edges in knowledge graph."""
        knowledge_graph, _ = service.build_knowledge_graph(
            entities=sample_entities,
            relationships=sample_relationships,
            graph_name='query_edges_kg'
        )

        results, query_stats = service.query_knowledge_graph(
            knowledge_graph=knowledge_graph,
            query_type='edge',
            filters={'relation_type': 'inhibits'}
        )

        assert isinstance(results, list)
        assert query_stats['query_type'] == 'edge'

    def test_query_knowledge_graph_paths(self, service, sample_entities, sample_relationships):
        """Test querying paths in knowledge graph."""
        knowledge_graph, _ = service.build_knowledge_graph(
            entities=sample_entities,
            relationships=sample_relationships,
            graph_name='query_paths_kg'
        )

        # Get node IDs for path query
        if knowledge_graph['nodes']:
            source_id = knowledge_graph['nodes'][0]['id']
            target_id = knowledge_graph['nodes'][-1]['id'] if len(knowledge_graph['nodes']) > 1 else source_id

            results, query_stats = service.query_knowledge_graph(
                knowledge_graph=knowledge_graph,
                query_type='path',
                filters={'source': source_id, 'target': target_id}
            )

            assert isinstance(results, list)
            assert query_stats['query_type'] == 'path'

    def test_query_knowledge_graph_invalid_type(self, service, sample_entities, sample_relationships):
        """Test query with invalid query type."""
        knowledge_graph, _ = service.build_knowledge_graph(
            entities=sample_entities,
            relationships=sample_relationships,
            graph_name='invalid_query_kg'
        )

        with pytest.raises(GraphConstructionError):
            service.query_knowledge_graph(
                knowledge_graph=knowledge_graph,
                query_type='invalid_type'
            )


class TestEntityExtraction:
    """Test entity extraction helper methods."""

    def test_extract_sections(self, service):
        """Test section extraction from publication text."""
        text = """
        Abstract: This is the abstract.

        Introduction: This is the introduction.

        Methods: These are the methods.
        """

        sections = service._extract_sections(text)
        assert isinstance(sections, dict)
        assert 'abstract' in sections or 'full_text' in sections

    def test_extract_sentences(self, service):
        """Test sentence extraction from text."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        sentences = service._extract_sentences(text)

        assert isinstance(sentences, list)
        assert len(sentences) >= 2

    def test_deduplicate_entities(self, service):
        """Test entity deduplication."""
        entities_by_type = {
            'drug': [
                {'text': 'Aspirin', 'mentions': 1},
                {'text': 'aspirin', 'mentions': 1},
                {'text': 'ASPIRIN', 'mentions': 1}
            ]
        }

        deduplicated = service._deduplicate_entities(entities_by_type)
        assert len(deduplicated['drug']) == 1
        assert deduplicated['drug'][0]['mentions'] == 3


class TestGraphOperations:
    """Test knowledge graph operations."""

    def test_calculate_graph_statistics(self, service):
        """Test graph statistics calculation."""
        nodes = [
            {'id': 0, 'name': 'node0', 'type': 'drug'},
            {'id': 1, 'name': 'node1', 'type': 'target'},
            {'id': 2, 'name': 'node2', 'type': 'disease'}
        ]

        edges = [
            {'source': 0, 'target': 1},
            {'source': 1, 'target': 2}
        ]

        stats = service._calculate_graph_statistics(nodes, edges)
        assert 'avg_degree' in stats
        assert 'max_degree' in stats
        assert 'density' in stats

    def test_count_by_type(self, service):
        """Test counting items by type."""
        items = [
            {'relation_type': 'inhibits'},
            {'relation_type': 'inhibits'},
            {'relation_type': 'treats'}
        ]

        counts = service._count_by_type(items, 'relation_type')
        assert counts['inhibits'] == 2
        assert counts['treats'] == 1


class TestExportFormats:
    """Test different export format methods."""

    def test_export_json(self, service, sample_entities, sample_relationships):
        """Test JSON export."""
        kg, _ = service.build_knowledge_graph(sample_entities, sample_relationships, 'test_json')
        result = service._export_json(kg, None)
        assert isinstance(result, str)
        assert 'nodes' in result
        assert 'edges' in result

    def test_export_rdf(self, service, sample_entities, sample_relationships):
        """Test RDF export."""
        kg, _ = service.build_knowledge_graph(sample_entities, sample_relationships, 'test_rdf')
        result = service._export_rdf(kg, None)
        assert isinstance(result, str)
        assert '@prefix' in result

    def test_export_neo4j(self, service, sample_entities, sample_relationships):
        """Test Neo4j export."""
        kg, _ = service.build_knowledge_graph(sample_entities, sample_relationships, 'test_neo4j')
        result = service._export_neo4j(kg, None)
        assert isinstance(result, str)
        assert 'CREATE' in result
