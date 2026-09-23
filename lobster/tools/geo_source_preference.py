"""Informational GEO source preference shared by research preparation tools."""

from langgraph.errors import GraphInterrupt
from langgraph.types import interrupt

from lobster.services.interaction.component_mapper import map_question
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def ask_geo_source_preference(accession: str) -> str:
    """Pause before queue creation. The answer does not change download execution."""
    logger.info("[GEO preference] question mapping start accession=%s", accession)
    options = ["Author-uploaded GEO data", "NCBI-generated counts"]
    selection = map_question(
        f"NCBI-generated counts are available for {accession}. Which source do you "
        "prefer? This preference is recorded only; the existing download method "
        "will still be used in this version.",
        {"options": options},
    )
    logger.info(
        "[GEO preference] interrupt requested accession=%s component=%s",
        accession,
        selection.component,
    )
    try:
        response = interrupt(selection.model_dump())
    except GraphInterrupt:
        logger.info(
            "[GEO preference] interrupt raised; awaiting user accession=%s", accession
        )
        raise
    except Exception:
        logger.exception("[GEO preference] interrupt failed accession=%s", accession)
        raise
    logger.info(
        "[GEO preference] interrupt resumed accession=%s response_type=%s",
        accession,
        type(response).__name__,
    )
    value = response.get("selected") if isinstance(response, dict) else response
    choices = {
        options[0]: "author",
        options[1]: "ncbi",
        "author": "author",
        "ncbi": "ncbi",
    }
    if value not in choices:
        logger.error(
            "[GEO preference] invalid answer accession=%s value_type=%s",
            accession,
            type(value).__name__,
        )
        raise ValueError(
            "No valid GEO source preference was supplied; entry was not created"
        )
    logger.info(
        "[GEO preference] answer accepted accession=%s selected_source=%s",
        accession,
        choices[value],
    )
    return choices[value]
