"""
Pharmacology Knowledge Graph Expert Agent for analyzing pharmacology publications and constructing knowledge graphs.

This agent focuses on parsing pharmacology publications, extracting pharmacological entities,
identifying relationships, and constructing knowledge graphs using the modular DataManagerV2 system.
"""

from datetime import date
from typing import Any, Dict, List, Optional
from pathlib import Path

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.state import PharmacologyKnowledgeGraphExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.pharmacology_knowledge_graph_service import (
    PharmacologyKnowledgeGraphService,
    PharmacologyKnowledgeGraphError,
    EntityExtractionError,
    GraphConstructionError
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class PharmacologyKGError(Exception):
    """Base exception for pharmacology knowledge graph operations."""
    pass


class PublicationNotFoundError(PharmacologyKGError):
    """Raised when publication is not found."""
    pass


def pharmacology_knowledge_graph_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "pharmacology_knowledge_graph_expert_agent",
    handoff_tools: List = None,
):
    """Create pharmacology knowledge graph expert agent using DataManagerV2."""

    settings = get_settings()
    model_params = settings.get_agent_llm_params("pharmacology_knowledge_graph_expert_agent")
    llm = create_llm("pharmacology_knowledge_graph_expert_agent", model_params)

    if callback_handler and hasattr(llm, "with_config"):
        llm = llm.with_config(callbacks=[callback_handler])

    # Initialize service
    service = PharmacologyKnowledgeGraphService()

    # Store knowledge graph results
    kg_results = {"summary": "", "details": {}, "knowledge_graphs": {}}

    # -------------------------
    # PUBLICATION PARSING TOOLS
    # -------------------------
    @tool
    def parse_pharmacology_publication(
        publication_text: str,
        publication_id: Optional[str] = None,
        source_type: str = "full_text",
        save_result: bool = True
    ) -> str:
        """
        Parse a pharmacology publication and extract structured content.

        Args:
            publication_text: Full text or abstract of the publication
            publication_id: Optional publication identifier (PMID, DOI)
            source_type: Type of source ("full_text", "abstract", "pdf")
            save_result: Whether to save parsed content

        Returns:
            str: Summary of parsing results
        """
        try:
            logger.info(f"Parsing pharmacology publication: {publication_id or 'unknown'}")

            # Call stateless service
            parsed_content, parsing_stats = service.parse_publication(
                publication_text=publication_text,
                publication_id=publication_id,
                source_type=source_type
            )

            # Store parsed content in data manager metadata
            storage_key = f"parsed_publication_{publication_id or 'latest'}"
            data_manager.metadata_store[storage_key] = parsed_content

            # Log operation
            data_manager.log_tool_usage(
                tool_name="parse_pharmacology_publication",
                parameters={
                    "publication_id": publication_id,
                    "source_type": source_type,
                    "text_length": len(publication_text)
                },
                description=f"Parsed publication: {parsing_stats['sentences_extracted']} sentences"
            )

            response = f"""✅ Successfully parsed pharmacology publication!

📄 **Publication:** {publication_id or 'Unknown ID'}

📊 **Parsing Results:**
- Source type: {source_type}
- Text length: {len(publication_text):,} characters
- Sections extracted: {parsing_stats['sections_extracted']}
- Sentences extracted: {parsing_stats['sentences_extracted']}

💾 **Storage:** Parsed content saved as '{storage_key}'

**Next Steps:** Use extract_pharmacology_entities() to identify drugs, targets, and diseases."""

            kg_results["details"]["publication_parsing"] = response
            return response

        except PharmacologyKnowledgeGraphError as e:
            logger.error(f"Error parsing publication: {e}")
            return f"❌ Error parsing publication: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in publication parsing: {e}")
            return f"❌ Unexpected error: {str(e)}"

    @tool
    def extract_pharmacology_entities(
        publication_id: str,
        entity_types: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
        save_result: bool = True
    ) -> str:
        """
        Extract pharmacological entities from parsed publication.

        Args:
            publication_id: Publication identifier to extract entities from
            entity_types: List of entity types to extract (drug, target, disease, mechanism, interaction, outcome)
            confidence_threshold: Minimum confidence score (0.0-1.0)
            save_result: Whether to save extracted entities

        Returns:
            str: Summary of entity extraction results
        """
        try:
            logger.info(f"Extracting entities from publication: {publication_id}")

            # Get parsed content
            storage_key = f"parsed_publication_{publication_id}"
            if storage_key not in data_manager.metadata_store:
                raise PublicationNotFoundError(
                    f"Publication '{publication_id}' not found. Parse it first using parse_pharmacology_publication()."
                )

            parsed_content = data_manager.metadata_store[storage_key]

            # Call stateless service
            entities, extraction_stats = service.extract_entities(
                parsed_content=parsed_content,
                entity_types=entity_types,
                confidence_threshold=confidence_threshold
            )

            # Store entities
            entities_key = f"entities_{publication_id}"
            data_manager.metadata_store[entities_key] = entities

            # Log operation
            data_manager.log_tool_usage(
                tool_name="extract_pharmacology_entities",
                parameters={
                    "publication_id": publication_id,
                    "entity_types": entity_types,
                    "confidence_threshold": confidence_threshold
                },
                description=f"Extracted {extraction_stats['total_entities']} entities"
            )

            # Format response
            response = f"""✅ Successfully extracted pharmacological entities!

📄 **Publication:** {publication_id}

📊 **Extraction Results:**
- Total entities: {extraction_stats['total_entities']}
- Confidence threshold: {confidence_threshold}

**Entities by Type:**
"""

            for entity_type, count in extraction_stats['entities_by_type'].items():
                response += f"- {entity_type.capitalize()}: {count} entities\n"

            response += f"""
💾 **Storage:** Entities saved as '{entities_key}'

**Next Steps:** Use extract_pharmacology_relationships() to identify relationships between entities."""

            kg_results["details"]["entity_extraction"] = response
            return response

        except PublicationNotFoundError as e:
            logger.error(f"Publication not found: {e}")
            return f"❌ {str(e)}"
        except EntityExtractionError as e:
            logger.error(f"Error extracting entities: {e}")
            return f"❌ Error extracting entities: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in entity extraction: {e}")
            return f"❌ Unexpected error: {str(e)}"

    @tool
    def extract_pharmacology_relationships(
        publication_id: str,
        relationship_types: Optional[List[str]] = None,
        save_result: bool = True
    ) -> str:
        """
        Extract relationships between pharmacological entities.

        Args:
            publication_id: Publication identifier
            relationship_types: List of relationship types to extract
            save_result: Whether to save extracted relationships

        Returns:
            str: Summary of relationship extraction results
        """
        try:
            logger.info(f"Extracting relationships from publication: {publication_id}")

            # Get parsed content and entities
            parsed_key = f"parsed_publication_{publication_id}"
            entities_key = f"entities_{publication_id}"

            if parsed_key not in data_manager.metadata_store:
                raise PublicationNotFoundError(
                    f"Publication '{publication_id}' not parsed. Parse it first."
                )

            if entities_key not in data_manager.metadata_store:
                raise PublicationNotFoundError(
                    f"Entities for '{publication_id}' not extracted. Extract entities first."
                )

            parsed_content = data_manager.metadata_store[parsed_key]
            entities = data_manager.metadata_store[entities_key]

            # Call stateless service
            relationships, extraction_stats = service.extract_relationships(
                parsed_content=parsed_content,
                entities=entities,
                relationship_types=relationship_types
            )

            # Store relationships
            relationships_key = f"relationships_{publication_id}"
            data_manager.metadata_store[relationships_key] = relationships

            # Log operation
            data_manager.log_tool_usage(
                tool_name="extract_pharmacology_relationships",
                parameters={
                    "publication_id": publication_id,
                    "relationship_types": relationship_types
                },
                description=f"Extracted {extraction_stats['total_relationships']} relationships"
            )

            # Format response
            response = f"""✅ Successfully extracted pharmacological relationships!

📄 **Publication:** {publication_id}

📊 **Extraction Results:**
- Total relationships: {extraction_stats['total_relationships']}
- Unique entity pairs: {extraction_stats['unique_entity_pairs']}

**Relationships by Type:**
"""

            for rel_type, count in extraction_stats['relationships_by_type'].items():
                response += f"- {rel_type}: {count}\n"

            response += f"""
💾 **Storage:** Relationships saved as '{relationships_key}'

**Next Steps:** Use build_pharmacology_knowledge_graph() to construct the knowledge graph."""

            kg_results["details"]["relationship_extraction"] = response
            return response

        except PublicationNotFoundError as e:
            logger.error(f"Data not found: {e}")
            return f"❌ {str(e)}"
        except EntityExtractionError as e:
            logger.error(f"Error extracting relationships: {e}")
            return f"❌ Error extracting relationships: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in relationship extraction: {e}")
            return f"❌ Unexpected error: {str(e)}"

    # -------------------------
    # KNOWLEDGE GRAPH CONSTRUCTION TOOLS
    # -------------------------
    @tool
    def build_pharmacology_knowledge_graph(
        publication_id: str,
        graph_name: Optional[str] = None,
        save_result: bool = True
    ) -> str:
        """
        Build a pharmacology knowledge graph from extracted entities and relationships.

        Args:
            publication_id: Publication identifier
            graph_name: Optional name for the knowledge graph
            save_result: Whether to save the knowledge graph

        Returns:
            str: Summary of knowledge graph construction
        """
        try:
            logger.info(f"Building knowledge graph from publication: {publication_id}")

            # Get entities and relationships
            entities_key = f"entities_{publication_id}"
            relationships_key = f"relationships_{publication_id}"

            if entities_key not in data_manager.metadata_store:
                raise PublicationNotFoundError(
                    f"Entities for '{publication_id}' not found. Extract entities first."
                )

            if relationships_key not in data_manager.metadata_store:
                raise PublicationNotFoundError(
                    f"Relationships for '{publication_id}' not found. Extract relationships first."
                )

            entities = data_manager.metadata_store[entities_key]
            relationships = data_manager.metadata_store[relationships_key]

            # Generate graph name
            graph_name = graph_name or f"kg_{publication_id}"

            # Call stateless service
            knowledge_graph, construction_stats = service.build_knowledge_graph(
                entities=entities,
                relationships=relationships,
                graph_name=graph_name
            )

            # Store knowledge graph
            kg_results["knowledge_graphs"][graph_name] = knowledge_graph
            data_manager.metadata_store[f"knowledge_graph_{graph_name}"] = knowledge_graph

            # Log operation
            data_manager.log_tool_usage(
                tool_name="build_pharmacology_knowledge_graph",
                parameters={
                    "publication_id": publication_id,
                    "graph_name": graph_name
                },
                description=f"Built knowledge graph: {construction_stats['n_nodes']} nodes, {construction_stats['n_edges']} edges"
            )

            # Format response
            response = f"""✅ Successfully built pharmacology knowledge graph!

🔗 **Graph Name:** {graph_name}

📊 **Graph Statistics:**
- Nodes: {construction_stats['n_nodes']}
- Edges: {construction_stats['n_edges']}
- Graph density: {construction_stats['density']:.4f}
- Average degree: {construction_stats['avg_degree']:.2f}
- Max degree: {construction_stats['max_degree']}

**Nodes by Type:**
"""

            for node_type, count in construction_stats['nodes_by_type'].items():
                response += f"- {node_type.capitalize()}: {count} nodes\n"

            response += "\n**Edges by Type:**\n"
            for edge_type, count in construction_stats['edges_by_type'].items():
                response += f"- {edge_type}: {count} edges\n"

            response += f"""
💾 **Storage:** Knowledge graph saved as '{graph_name}'

**Next Steps:**
- Use visualize_pharmacology_knowledge_graph() to create an interactive visualization
- Use export_pharmacology_knowledge_graph() to export in various formats
- Use query_pharmacology_knowledge_graph() to query the graph"""

            kg_results["details"]["knowledge_graph_construction"] = response
            return response

        except PublicationNotFoundError as e:
            logger.error(f"Data not found: {e}")
            return f"❌ {str(e)}"
        except GraphConstructionError as e:
            logger.error(f"Error building knowledge graph: {e}")
            return f"❌ Error building knowledge graph: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in knowledge graph construction: {e}")
            return f"❌ Unexpected error: {str(e)}"

    @tool
    def visualize_pharmacology_knowledge_graph(
        graph_name: str,
        layout: str = "force",
        node_size_by: str = "mentions",
        color_by: str = "type",
        save_result: bool = True
    ) -> str:
        """
        Create an interactive visualization of the pharmacology knowledge graph.

        Args:
            graph_name: Name of the knowledge graph to visualize
            layout: Graph layout algorithm ("force", "circular", "hierarchical")
            node_size_by: Node attribute to determine size
            color_by: Node attribute for coloring
            save_result: Whether to save the visualization

        Returns:
            str: Summary of visualization creation
        """
        try:
            logger.info(f"Visualizing knowledge graph: {graph_name}")

            # Get knowledge graph
            kg_key = f"knowledge_graph_{graph_name}"
            if kg_key not in data_manager.metadata_store:
                available_graphs = [k.replace('knowledge_graph_', '') for k in data_manager.metadata_store.keys() if k.startswith('knowledge_graph_')]
                return f"❌ Knowledge graph '{graph_name}' not found. Available graphs: {available_graphs}"

            knowledge_graph = data_manager.metadata_store[kg_key]

            # Call stateless service
            fig, viz_stats = service.visualize_knowledge_graph(
                knowledge_graph=knowledge_graph,
                layout=layout,
                node_size_by=node_size_by,
                color_by=color_by
            )

            # Save visualization
            plot_title = f"Pharmacology Knowledge Graph: {graph_name}"
            data_manager.add_plot(
                fig,
                title=plot_title,
                source="pharmacology_knowledge_graph_service",
                dataset_info={
                    "graph_name": graph_name,
                    "n_nodes": viz_stats['n_nodes_displayed'],
                    "n_edges": viz_stats['n_edges_displayed']
                },
                analysis_params={
                    "layout": layout,
                    "node_size_by": node_size_by,
                    "color_by": color_by
                }
            )

            # Log operation
            data_manager.log_tool_usage(
                tool_name="visualize_pharmacology_knowledge_graph",
                parameters={
                    "graph_name": graph_name,
                    "layout": layout,
                    "node_size_by": node_size_by,
                    "color_by": color_by
                },
                description=f"Created visualization for {graph_name}"
            )

            response = f"""✅ Successfully created knowledge graph visualization!

🔗 **Graph:** {graph_name}

📊 **Visualization:**
- Layout: {layout}
- Nodes displayed: {viz_stats['n_nodes_displayed']}
- Edges displayed: {viz_stats['n_edges_displayed']}
- Node size by: {node_size_by}
- Color by: {color_by}

📈 **Plot saved:** {plot_title}

**Interactive Features:**
- Hover over nodes to see details
- Zoom and pan to explore
- Node colors represent entity types
- Node sizes represent {node_size_by}

Use /plots to view the interactive visualization."""

            kg_results["details"]["visualization"] = response
            return response

        except GraphConstructionError as e:
            logger.error(f"Error visualizing knowledge graph: {e}")
            return f"❌ Error visualizing knowledge graph: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in visualization: {e}")
            return f"❌ Unexpected error: {str(e)}"

    @tool
    def export_pharmacology_knowledge_graph(
        graph_name: str,
        export_format: str = "json",
        output_filename: Optional[str] = None
    ) -> str:
        """
        Export pharmacology knowledge graph in various formats.

        Args:
            graph_name: Name of the knowledge graph to export
            export_format: Export format ("json", "rdf", "neo4j", "networkx", "graphml")
            output_filename: Optional output filename

        Returns:
            str: Summary of export operation
        """
        try:
            logger.info(f"Exporting knowledge graph: {graph_name} as {export_format}")

            # Get knowledge graph
            kg_key = f"knowledge_graph_{graph_name}"
            if kg_key not in data_manager.metadata_store:
                available_graphs = [k.replace('knowledge_graph_', '') for k in data_manager.metadata_store.keys() if k.startswith('knowledge_graph_')]
                return f"❌ Knowledge graph '{graph_name}' not found. Available graphs: {available_graphs}"

            knowledge_graph = data_manager.metadata_store[kg_key]

            # Determine output path
            output_path = None
            if output_filename:
                output_path = data_manager.exports_dir / output_filename

            # Call stateless service
            export_result, export_stats = service.export_knowledge_graph(
                knowledge_graph=knowledge_graph,
                export_format=export_format,
                output_path=output_path
            )

            # Log operation
            data_manager.log_tool_usage(
                tool_name="export_pharmacology_knowledge_graph",
                parameters={
                    "graph_name": graph_name,
                    "export_format": export_format,
                    "output_filename": output_filename
                },
                description=f"Exported {graph_name} as {export_format}"
            )

            response = f"""✅ Successfully exported knowledge graph!

🔗 **Graph:** {graph_name}

📊 **Export Details:**
- Format: {export_format}
- Nodes: {export_stats['n_nodes']}
- Edges: {export_stats['n_edges']}
- Output: {export_stats['output_path']}

**Format Details:**
"""

            if export_format == "json":
                response += "- Standard JSON format for general use\n- Compatible with web applications\n"
            elif export_format == "rdf":
                response += "- RDF/Turtle format for semantic web\n- Can be loaded into triple stores\n"
            elif export_format == "neo4j":
                response += "- Cypher queries for Neo4j graph database\n- Ready to execute in Neo4j\n"
            elif export_format == "networkx":
                response += "- Edge list format for NetworkX\n- Compatible with Python graph analysis\n"
            elif export_format == "graphml":
                response += "- GraphML format for general graph tools\n- Compatible with Gephi, Cytoscape\n"

            if output_path:
                response += f"\n📁 **File saved:** {output_path}"
            else:
                response += f"\n📄 **Result:** Export available in memory (not saved to file)"

            kg_results["details"]["export"] = response
            return response

        except GraphConstructionError as e:
            logger.error(f"Error exporting knowledge graph: {e}")
            return f"❌ Error exporting knowledge graph: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in export: {e}")
            return f"❌ Unexpected error: {str(e)}"

    @tool
    def query_pharmacology_knowledge_graph(
        graph_name: str,
        query_type: str = "node",
        filters: Optional[str] = None
    ) -> str:
        """
        Query pharmacology knowledge graph for specific nodes, edges, or patterns.

        Args:
            graph_name: Name of the knowledge graph to query
            query_type: Type of query ("node", "edge", "path", "subgraph")
            filters: JSON string of query filters (e.g., '{"type": "drug", "name": "aspirin"}')

        Returns:
            str: Summary of query results
        """
        try:
            import json

            logger.info(f"Querying knowledge graph: {graph_name}")

            # Get knowledge graph
            kg_key = f"knowledge_graph_{graph_name}"
            if kg_key not in data_manager.metadata_store:
                available_graphs = [k.replace('knowledge_graph_', '') for k in data_manager.metadata_store.keys() if k.startswith('knowledge_graph_')]
                return f"❌ Knowledge graph '{graph_name}' not found. Available graphs: {available_graphs}"

            knowledge_graph = data_manager.metadata_store[kg_key]

            # Parse filters
            filter_dict = json.loads(filters) if filters else {}

            # Call stateless service
            results, query_stats = service.query_knowledge_graph(
                knowledge_graph=knowledge_graph,
                query_type=query_type,
                filters=filter_dict
            )

            # Log operation
            data_manager.log_tool_usage(
                tool_name="query_pharmacology_knowledge_graph",
                parameters={
                    "graph_name": graph_name,
                    "query_type": query_type,
                    "filters": filter_dict
                },
                description=f"Query returned {query_stats['n_results']} results"
            )

            # Format response
            response = f"""✅ Query completed successfully!

🔗 **Graph:** {graph_name}

📊 **Query:**
- Type: {query_type}
- Filters: {filter_dict}
- Results: {query_stats['n_results']}

**Results Preview:**
"""

            # Show preview of results (limit to 10)
            for i, result in enumerate(results[:10]):
                if query_type == "node":
                    response += f"{i+1}. {result.get('name', 'Unknown')} (type: {result.get('type', 'Unknown')})\n"
                elif query_type == "edge":
                    response += f"{i+1}. {result.get('relation_type', 'Unknown')} (confidence: {result.get('confidence', 0):.2f})\n"
                elif query_type == "path":
                    response += f"{i+1}. Path with {len(result.get('nodes', []))} nodes\n"
                elif query_type == "subgraph":
                    response += f"{i+1}. Subgraph: {len(result.get('nodes', []))} nodes, {len(result.get('edges', []))} edges\n"

            if len(results) > 10:
                response += f"\n... and {len(results) - 10} more results"

            kg_results["details"]["query"] = response
            return response

        except json.JSONDecodeError as e:
            logger.error(f"Invalid filters JSON: {e}")
            return f"❌ Invalid filters JSON: {str(e)}. Use format: '{\"type\": \"drug\"}'"
        except GraphConstructionError as e:
            logger.error(f"Error querying knowledge graph: {e}")
            return f"❌ Error querying knowledge graph: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in query: {e}")
            return f"❌ Unexpected error: {str(e)}"

    @tool
    def create_pharmacology_kg_summary() -> str:
        """Create a comprehensive summary of all pharmacology knowledge graph operations performed."""
        try:
            if not kg_results["details"]:
                return "No pharmacology knowledge graph operations have been performed yet. Start by parsing a publication."

            summary = "# Pharmacology Knowledge Graph Analysis Summary\n\n"

            for step, details in kg_results["details"].items():
                summary += f"## {step.replace('_', ' ').title()}\n"
                summary += f"{details}\n\n"

            # Add knowledge graph inventory
            if kg_results["knowledge_graphs"]:
                summary += f"## Available Knowledge Graphs\n"
                summary += f"Total graphs: {len(kg_results['knowledge_graphs'])}\n\n"

                for graph_name, graph_data in kg_results["knowledge_graphs"].items():
                    metadata = graph_data.get('metadata', {})
                    summary += f"### {graph_name}\n"
                    summary += f"- Nodes: {metadata.get('n_nodes', 0)}\n"
                    summary += f"- Edges: {metadata.get('n_edges', 0)}\n"
                    summary += f"- Created: {metadata.get('created_at', 'Unknown')}\n\n"

            kg_results["summary"] = summary
            logger.info("Created pharmacology knowledge graph summary")
            return summary

        except Exception as e:
            logger.error(f"Error creating summary: {e}")
            return f"❌ Error creating summary: {str(e)}"

    # -------------------------
    # TOOL REGISTRY
    # -------------------------
    base_tools = [
        parse_pharmacology_publication,
        extract_pharmacology_entities,
        extract_pharmacology_relationships,
        build_pharmacology_knowledge_graph,
        visualize_pharmacology_knowledge_graph,
        export_pharmacology_knowledge_graph,
        query_pharmacology_knowledge_graph,
        create_pharmacology_kg_summary,
    ]

    tools = base_tools + (handoff_tools or [])

    # -------------------------
    # SYSTEM PROMPT
    # -------------------------
    system_prompt = f"""
You are an expert pharmacology knowledge graph specialist focusing on analyzing pharmacology publications and constructing comprehensive knowledge graphs using the professional, modular DataManagerV2 system.

<Role>
You parse pharmacology publications, extract pharmacological entities (drugs, targets, diseases, mechanisms), identify relationships, and construct knowledge graphs for drug discovery and pharmacological research. You work with structured knowledge representation with full provenance tracking and professional-grade error handling.

**CRITICAL: You ONLY perform pharmacology knowledge graph tasks specifically requested by the supervisor. You report results back to the supervisor, never directly to users.**
</Role>

<Communication Flow>
**USER → SUPERVISOR → YOU → SUPERVISOR → USER**
- You receive pharmacology knowledge graph tasks from the supervisor
- You execute the requested analysis and graph construction
- You report results back to the supervisor
- The supervisor communicates with the user
</Communication Flow>

<Task>
You perform pharmacology knowledge graph analysis following best practices:
1. **Publication parsing** - parse PubMed abstracts, DOIs, PMIDs, or full texts
2. **Entity extraction** - identify drugs, targets, diseases, mechanisms, interactions, outcomes
3. **Relationship extraction** - identify relationships between entities (inhibits, treats, etc.)
4. **Knowledge graph construction** - build structured graphs from entities and relationships
5. **Visualization** - create interactive network visualizations
6. **Export** - export graphs in multiple formats (JSON, RDF, Neo4j, NetworkX, GraphML)
7. **Query** - query graphs for specific patterns and relationships
8. **Comprehensive reporting** - document all operations with provenance tracking
</Task>

<Available Tools>

## Publication Parsing:
- `parse_pharmacology_publication`: Parse publication text and extract structured content
- Entity extraction requires parsed content

## Entity and Relationship Extraction:
- `extract_pharmacology_entities`: Extract pharmacological entities (drug, target, disease, mechanism, interaction, outcome)
- `extract_pharmacology_relationships`: Identify relationships between entities
- Uses NLP and text mining techniques

## Knowledge Graph Construction:
- `build_pharmacology_knowledge_graph`: Construct knowledge graph from entities and relationships
- Creates nodes and edges with properties
- Calculates graph statistics

## Visualization and Export:
- `visualize_pharmacology_knowledge_graph`: Create interactive network visualizations
- `export_pharmacology_knowledge_graph`: Export in multiple formats (JSON, RDF, Neo4j, NetworkX, GraphML)
- `query_pharmacology_knowledge_graph`: Query for nodes, edges, paths, or subgraphs

## Summary:
- `create_pharmacology_kg_summary`: Generate comprehensive analysis summary

</Available Tools>

<Professional Workflows & Tool Usage Order>

## 1. COMPLETE KNOWLEDGE GRAPH PIPELINE (Supervisor: "Create knowledge graph from publication")

### Full Pipeline

# Step 1: Parse publication
parse_pharmacology_publication(
    publication_text="[full text or abstract]",
    publication_id="PMID12345",
    source_type="full_text"
)

# Step 2: Extract entities
extract_pharmacology_entities(
    publication_id="PMID12345",
    entity_types=["drug", "target", "disease"],
    confidence_threshold=0.5
)

# Step 3: Extract relationships
extract_pharmacology_relationships(
    publication_id="PMID12345",
    relationship_types=["inhibits", "treats", "activates"]
)

# Step 4: Build knowledge graph
build_pharmacology_knowledge_graph(
    publication_id="PMID12345",
    graph_name="pharmacology_kg_pmid12345"
)

# Step 5: Visualize
visualize_pharmacology_knowledge_graph(
    graph_name="pharmacology_kg_pmid12345",
    layout="force",
    node_size_by="mentions",
    color_by="type"
)

# Step 6: Export
export_pharmacology_knowledge_graph(
    graph_name="pharmacology_kg_pmid12345",
    export_format="json",
    output_filename="kg_pmid12345.json"
)

# Step 7: Report to supervisor


## 2. ENTITY-FOCUSED ANALYSIS (Supervisor: "Extract drugs and targets")

# Step 1: Parse publication
parse_pharmacology_publication(...)

# Step 2: Extract specific entity types
extract_pharmacology_entities(
    publication_id="PMID12345",
    entity_types=["drug", "target"],
    confidence_threshold=0.6
)

# Step 3: Report entities to supervisor


## 3. RELATIONSHIP ANALYSIS (Supervisor: "Find drug-target interactions")

# Step 1: Ensure publication is parsed and entities extracted
# Step 2: Extract specific relationships
extract_pharmacology_relationships(
    publication_id="PMID12345",
    relationship_types=["inhibits", "activates", "binds_to"]
)

# Step 3: Build graph for visualization
build_pharmacology_knowledge_graph(...)

# Step 4: Report to supervisor


## 4. MULTI-FORMAT EXPORT (Supervisor: "Export graph for Neo4j and NetworkX")

# Export for Neo4j
export_pharmacology_knowledge_graph(
    graph_name="kg_name",
    export_format="neo4j",
    output_filename="kg_neo4j.cypher"
)

# Export for NetworkX
export_pharmacology_knowledge_graph(
    graph_name="kg_name",
    export_format="networkx",
    output_filename="kg_networkx.txt"
)


## 5. QUERY AND EXPLORATION (Supervisor: "Find all drugs targeting protein X")

# Query for specific nodes
query_pharmacology_knowledge_graph(
    graph_name="kg_name",
    query_type="node",
    filters='{"type": "drug"}'
)

# Query for specific relationships
query_pharmacology_knowledge_graph(
    graph_name="kg_name",
    query_type="edge",
    filters='{"relation_type": "inhibits"}'
)

# Find paths between entities
query_pharmacology_knowledge_graph(
    graph_name="kg_name",
    query_type="path",
    filters='{"source": "drug_name", "target": "protein_name"}'
)

</Professional Workflows & Tool Usage Order>

<Entity Types>
- **drug**: Drugs, compounds, molecules, medications, pharmaceuticals
- **target**: Proteins, genes, receptors, enzymes, kinases, channels
- **disease**: Diseases, disorders, syndromes, conditions, pathologies
- **mechanism**: Inhibition, activation, modulation, agonism, antagonism
- **interaction**: Drug-drug interactions, synergy, antagonism
- **outcome**: Efficacy, toxicity, adverse effects, clinical outcomes
</Entity Types>

<Relationship Types>
- inhibits, activates, binds_to
- treats, indicated_for, contraindicated
- interacts_with, metabolizes, targets
- associated_with, causes, prevents
</Relationship Types>

<Export Formats>
- **json**: Standard JSON for general use and web applications
- **rdf**: RDF/Turtle for semantic web and triple stores
- **neo4j**: Cypher queries for Neo4j graph database
- **networkx**: Edge list for NetworkX Python library
- **graphml**: GraphML for Gephi, Cytoscape, and other graph tools
</Export Formats>

<Critical Operating Principles>
1. **ONLY perform tasks explicitly requested by the supervisor**
2. **Always report results back to the supervisor, never directly to users**
3. **Validate data existence before operations** (parsed content, entities, relationships)
4. **Store all results in data manager** for provenance tracking
5. **Use descriptive storage keys** (e.g., "entities_PMID12345")
6. **Follow the pipeline order** (parse → extract entities → extract relationships → build graph)
7. **Provide clear error messages** when data is missing
8. **Wait for supervisor instruction** between major analysis steps
9. **Document all operations** for reproducibility
10. **NEVER HALLUCINATE OR LIE** - only report operations actually completed

</Critical Operating Principles>

<Quality Assurance & Best Practices>
- All tools include professional error handling with specific exception types
- Comprehensive logging tracks all operations with parameters
- Automatic validation ensures data existence throughout pipeline
- Provenance tracking maintains complete analysis history
- Professional reporting with statistical summaries
- Support for batch processing multiple publications
- Integration with existing bioinformatics workflows

</Quality Assurance & Best Practices>

Today's date: {date.today()}
""".strip()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=PharmacologyKnowledgeGraphExpertState,
    )
