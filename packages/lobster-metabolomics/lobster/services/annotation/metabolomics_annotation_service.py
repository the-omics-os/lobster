"""
Metabolomics annotation service for m/z-based metabolite identification
and lipid class classification.

This service implements metabolite annotation by matching observed m/z values
against a bundled reference database of common metabolites with adduct correction,
ppm tolerance matching, and MSI confidence level assignment.

All methods return 3-tuples (AnnData, Dict, AnalysisStep) for provenance tracking and
reproducible notebook export via /pipeline export.
"""

from typing import Any, Dict, List, Optional, Tuple

import anndata
import numpy as np

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Bundled reference database (~80 common metabolites for v1)
# Structure: {name: {"monoisotopic_mass": float, "formula": str,
#             "kegg_id": str, "hmdb_id": str, "class": str}}
# =============================================================================

METABOLITE_REFERENCE_DB: Dict[str, Dict[str, Any]] = {
    # Amino acids (20 standard)
    "Glycine": {
        "monoisotopic_mass": 75.03203,
        "formula": "C2H5NO2",
        "kegg_id": "C00037",
        "hmdb_id": "HMDB0000123",
        "class": "amino_acid",
    },
    "Alanine": {
        "monoisotopic_mass": 89.04768,
        "formula": "C3H7NO2",
        "kegg_id": "C00041",
        "hmdb_id": "HMDB0000161",
        "class": "amino_acid",
    },
    "Valine": {
        "monoisotopic_mass": 117.07898,
        "formula": "C5H11NO2",
        "kegg_id": "C00183",
        "hmdb_id": "HMDB0000883",
        "class": "amino_acid",
    },
    "Leucine": {
        "monoisotopic_mass": 131.09463,
        "formula": "C6H13NO2",
        "kegg_id": "C00123",
        "hmdb_id": "HMDB0000687",
        "class": "amino_acid",
    },
    "Isoleucine": {
        "monoisotopic_mass": 131.09463,
        "formula": "C6H13NO2",
        "kegg_id": "C00407",
        "hmdb_id": "HMDB0000172",
        "class": "amino_acid",
    },
    "Proline": {
        "monoisotopic_mass": 115.06333,
        "formula": "C5H9NO2",
        "kegg_id": "C00148",
        "hmdb_id": "HMDB0000162",
        "class": "amino_acid",
    },
    "Phenylalanine": {
        "monoisotopic_mass": 165.07898,
        "formula": "C9H11NO2",
        "kegg_id": "C00079",
        "hmdb_id": "HMDB0000159",
        "class": "amino_acid",
    },
    "Tryptophan": {
        "monoisotopic_mass": 204.08988,
        "formula": "C11H12N2O2",
        "kegg_id": "C00078",
        "hmdb_id": "HMDB0000929",
        "class": "amino_acid",
    },
    "Methionine": {
        "monoisotopic_mass": 149.05105,
        "formula": "C5H11NO2S",
        "kegg_id": "C00073",
        "hmdb_id": "HMDB0000696",
        "class": "amino_acid",
    },
    "Serine": {
        "monoisotopic_mass": 105.04259,
        "formula": "C3H7NO3",
        "kegg_id": "C00065",
        "hmdb_id": "HMDB0000187",
        "class": "amino_acid",
    },
    "Threonine": {
        "monoisotopic_mass": 119.05824,
        "formula": "C4H9NO3",
        "kegg_id": "C00188",
        "hmdb_id": "HMDB0000167",
        "class": "amino_acid",
    },
    "Cysteine": {
        "monoisotopic_mass": 121.01975,
        "formula": "C3H7NO2S",
        "kegg_id": "C00097",
        "hmdb_id": "HMDB0000574",
        "class": "amino_acid",
    },
    "Tyrosine": {
        "monoisotopic_mass": 181.07389,
        "formula": "C9H11NO3",
        "kegg_id": "C00082",
        "hmdb_id": "HMDB0000158",
        "class": "amino_acid",
    },
    "Asparagine": {
        "monoisotopic_mass": 132.05349,
        "formula": "C4H8N2O3",
        "kegg_id": "C00152",
        "hmdb_id": "HMDB0000168",
        "class": "amino_acid",
    },
    "Glutamine": {
        "monoisotopic_mass": 146.06914,
        "formula": "C5H10N2O3",
        "kegg_id": "C00064",
        "hmdb_id": "HMDB0000641",
        "class": "amino_acid",
    },
    "Aspartic acid": {
        "monoisotopic_mass": 133.03751,
        "formula": "C4H7NO4",
        "kegg_id": "C00049",
        "hmdb_id": "HMDB0000191",
        "class": "amino_acid",
    },
    "Glutamic acid": {
        "monoisotopic_mass": 147.05316,
        "formula": "C5H9NO4",
        "kegg_id": "C00025",
        "hmdb_id": "HMDB0000148",
        "class": "amino_acid",
    },
    "Lysine": {
        "monoisotopic_mass": 146.10553,
        "formula": "C6H14N2O2",
        "kegg_id": "C00047",
        "hmdb_id": "HMDB0000182",
        "class": "amino_acid",
    },
    "Arginine": {
        "monoisotopic_mass": 174.11168,
        "formula": "C6H14N4O2",
        "kegg_id": "C00062",
        "hmdb_id": "HMDB0000517",
        "class": "amino_acid",
    },
    "Histidine": {
        "monoisotopic_mass": 155.06948,
        "formula": "C6H9N3O2",
        "kegg_id": "C00135",
        "hmdb_id": "HMDB0000177",
        "class": "amino_acid",
    },
    # Organic acids
    "Citric acid": {
        "monoisotopic_mass": 192.02700,
        "formula": "C6H8O7",
        "kegg_id": "C00158",
        "hmdb_id": "HMDB0000094",
        "class": "organic_acid",
    },
    "Lactic acid": {
        "monoisotopic_mass": 90.03169,
        "formula": "C3H6O3",
        "kegg_id": "C00186",
        "hmdb_id": "HMDB0000190",
        "class": "organic_acid",
    },
    "Pyruvic acid": {
        "monoisotopic_mass": 88.01604,
        "formula": "C3H4O3",
        "kegg_id": "C00022",
        "hmdb_id": "HMDB0000243",
        "class": "organic_acid",
    },
    "Succinic acid": {
        "monoisotopic_mass": 118.02661,
        "formula": "C4H6O4",
        "kegg_id": "C00042",
        "hmdb_id": "HMDB0000254",
        "class": "organic_acid",
    },
    "Fumaric acid": {
        "monoisotopic_mass": 116.01096,
        "formula": "C4H4O4",
        "kegg_id": "C00122",
        "hmdb_id": "HMDB0000134",
        "class": "organic_acid",
    },
    "Malic acid": {
        "monoisotopic_mass": 134.02153,
        "formula": "C4H6O5",
        "kegg_id": "C00149",
        "hmdb_id": "HMDB0000156",
        "class": "organic_acid",
    },
    "Oxaloacetic acid": {
        "monoisotopic_mass": 132.00588,
        "formula": "C4H4O5",
        "kegg_id": "C00036",
        "hmdb_id": "HMDB0000223",
        "class": "organic_acid",
    },
    "alpha-Ketoglutaric acid": {
        "monoisotopic_mass": 146.02153,
        "formula": "C5H6O5",
        "kegg_id": "C00026",
        "hmdb_id": "HMDB0000208",
        "class": "organic_acid",
    },
    "Acetic acid": {
        "monoisotopic_mass": 60.02113,
        "formula": "C2H4O2",
        "kegg_id": "C00033",
        "hmdb_id": "HMDB0000042",
        "class": "organic_acid",
    },
    "Formic acid": {
        "monoisotopic_mass": 46.00548,
        "formula": "CH2O2",
        "kegg_id": "C00058",
        "hmdb_id": "HMDB0000142",
        "class": "organic_acid",
    },
    # Sugars
    "Glucose": {
        "monoisotopic_mass": 180.06339,
        "formula": "C6H12O6",
        "kegg_id": "C00031",
        "hmdb_id": "HMDB0000122",
        "class": "sugar",
    },
    "Fructose": {
        "monoisotopic_mass": 180.06339,
        "formula": "C6H12O6",
        "kegg_id": "C00095",
        "hmdb_id": "HMDB0000660",
        "class": "sugar",
    },
    "Galactose": {
        "monoisotopic_mass": 180.06339,
        "formula": "C6H12O6",
        "kegg_id": "C00124",
        "hmdb_id": "HMDB0000143",
        "class": "sugar",
    },
    "Sucrose": {
        "monoisotopic_mass": 342.11621,
        "formula": "C12H22O11",
        "kegg_id": "C00089",
        "hmdb_id": "HMDB0000258",
        "class": "sugar",
    },
    "Maltose": {
        "monoisotopic_mass": 342.11621,
        "formula": "C12H22O11",
        "kegg_id": "C00208",
        "hmdb_id": "HMDB0000163",
        "class": "sugar",
    },
    "Ribose": {
        "monoisotopic_mass": 150.05283,
        "formula": "C5H10O5",
        "kegg_id": "C00121",
        "hmdb_id": "HMDB0000283",
        "class": "sugar",
    },
    # Nucleotides and related
    "Adenine": {
        "monoisotopic_mass": 135.05450,
        "formula": "C5H5N5",
        "kegg_id": "C00147",
        "hmdb_id": "HMDB0000034",
        "class": "nucleotide",
    },
    "Guanine": {
        "monoisotopic_mass": 151.04940,
        "formula": "C5H5N5O",
        "kegg_id": "C00242",
        "hmdb_id": "HMDB0000132",
        "class": "nucleotide",
    },
    "Cytosine": {
        "monoisotopic_mass": 111.04326,
        "formula": "C4H5N3O",
        "kegg_id": "C00380",
        "hmdb_id": "HMDB0000630",
        "class": "nucleotide",
    },
    "Uracil": {
        "monoisotopic_mass": 112.02728,
        "formula": "C4H4N2O2",
        "kegg_id": "C00106",
        "hmdb_id": "HMDB0000300",
        "class": "nucleotide",
    },
    "Thymine": {
        "monoisotopic_mass": 126.04293,
        "formula": "C5H6N2O2",
        "kegg_id": "C00178",
        "hmdb_id": "HMDB0000262",
        "class": "nucleotide",
    },
    "ATP": {
        "monoisotopic_mass": 507.00011,
        "formula": "C10H16N5O13P3",
        "kegg_id": "C00002",
        "hmdb_id": "HMDB0000538",
        "class": "nucleotide",
    },
    "ADP": {
        "monoisotopic_mass": 427.02942,
        "formula": "C10H15N5O10P2",
        "kegg_id": "C00008",
        "hmdb_id": "HMDB0001341",
        "class": "nucleotide",
    },
    "AMP": {
        "monoisotopic_mass": 347.06308,
        "formula": "C10H14N5O7P",
        "kegg_id": "C00020",
        "hmdb_id": "HMDB0000045",
        "class": "nucleotide",
    },
    "NAD+": {
        "monoisotopic_mass": 663.10912,
        "formula": "C21H27N7O14P2",
        "kegg_id": "C00003",
        "hmdb_id": "HMDB0000902",
        "class": "nucleotide",
    },
    # Fatty acids
    "Palmitic acid": {
        "monoisotopic_mass": 256.24023,
        "formula": "C16H32O2",
        "kegg_id": "C00249",
        "hmdb_id": "HMDB0000220",
        "class": "fatty_acid",
    },
    "Stearic acid": {
        "monoisotopic_mass": 284.27153,
        "formula": "C18H36O2",
        "kegg_id": "C01530",
        "hmdb_id": "HMDB0000827",
        "class": "fatty_acid",
    },
    "Oleic acid": {
        "monoisotopic_mass": 282.25588,
        "formula": "C18H34O2",
        "kegg_id": "C00712",
        "hmdb_id": "HMDB0000207",
        "class": "fatty_acid",
    },
    "Linoleic acid": {
        "monoisotopic_mass": 280.24023,
        "formula": "C18H32O2",
        "kegg_id": "C01595",
        "hmdb_id": "HMDB0000673",
        "class": "fatty_acid",
    },
    "Arachidonic acid": {
        "monoisotopic_mass": 304.24023,
        "formula": "C20H32O2",
        "kegg_id": "C00219",
        "hmdb_id": "HMDB0001043",
        "class": "fatty_acid",
    },
    "Myristic acid": {
        "monoisotopic_mass": 228.20893,
        "formula": "C14H28O2",
        "kegg_id": "C06424",
        "hmdb_id": "HMDB0000806",
        "class": "fatty_acid",
    },
    "Lauric acid": {
        "monoisotopic_mass": 200.17763,
        "formula": "C12H24O2",
        "kegg_id": "C02679",
        "hmdb_id": "HMDB0000638",
        "class": "fatty_acid",
    },
    # Lipid classes (representative molecules)
    "LysoPC(16:0)": {
        "monoisotopic_mass": 495.33168,
        "formula": "C24H50NO7P",
        "kegg_id": "C04230",
        "hmdb_id": "HMDB0010382",
        "class": "lysophospholipid",
    },
    "LysoPC(18:0)": {
        "monoisotopic_mass": 523.36298,
        "formula": "C26H54NO7P",
        "kegg_id": "C04230",
        "hmdb_id": "HMDB0010384",
        "class": "lysophospholipid",
    },
    "LysoPC(18:1)": {
        "monoisotopic_mass": 521.34733,
        "formula": "C26H52NO7P",
        "kegg_id": "C04230",
        "hmdb_id": "HMDB0002815",
        "class": "lysophospholipid",
    },
    "PC(16:0/18:1)": {
        "monoisotopic_mass": 759.57764,
        "formula": "C42H82NO8P",
        "kegg_id": "C00157",
        "hmdb_id": "HMDB0000564",
        "class": "phospholipid",
    },
    "PC(16:0/18:2)": {
        "monoisotopic_mass": 757.56199,
        "formula": "C42H80NO8P",
        "kegg_id": "C00157",
        "hmdb_id": "HMDB0007973",
        "class": "phospholipid",
    },
    "PE(16:0/18:1)": {
        "monoisotopic_mass": 717.53069,
        "formula": "C39H76NO8P",
        "kegg_id": "C00350",
        "hmdb_id": "HMDB0009783",
        "class": "phospholipid",
    },
    "SM(d18:1/16:0)": {
        "monoisotopic_mass": 702.56692,
        "formula": "C39H79N2O6P",
        "kegg_id": "C00550",
        "hmdb_id": "HMDB0010168",
        "class": "sphingolipid",
    },
    "Ceramide(d18:1/16:0)": {
        "monoisotopic_mass": 537.51209,
        "formula": "C34H67NO3",
        "kegg_id": "C00195",
        "hmdb_id": "HMDB0004949",
        "class": "sphingolipid",
    },
    "TG(16:0/18:1/18:1)": {
        "monoisotopic_mass": 858.75680,
        "formula": "C55H102O6",
        "kegg_id": "C00422",
        "hmdb_id": "HMDB0005369",
        "class": "glycerolipid",
    },
    # Other common metabolites
    "Creatine": {
        "monoisotopic_mass": 131.06948,
        "formula": "C4H9N3O2",
        "kegg_id": "C00300",
        "hmdb_id": "HMDB0000064",
        "class": "other",
    },
    "Creatinine": {
        "monoisotopic_mass": 113.05891,
        "formula": "C4H7N3O",
        "kegg_id": "C00791",
        "hmdb_id": "HMDB0000562",
        "class": "other",
    },
    "Urea": {
        "monoisotopic_mass": 60.03236,
        "formula": "CH4N2O",
        "kegg_id": "C00086",
        "hmdb_id": "HMDB0000294",
        "class": "other",
    },
    "Uric acid": {
        "monoisotopic_mass": 168.02834,
        "formula": "C5H4N4O3",
        "kegg_id": "C00366",
        "hmdb_id": "HMDB0000289",
        "class": "other",
    },
    "Taurine": {
        "monoisotopic_mass": 125.01466,
        "formula": "C2H7NO3S",
        "kegg_id": "C00245",
        "hmdb_id": "HMDB0000251",
        "class": "amino_acid",
    },
    "Choline": {
        "monoisotopic_mass": 103.09971,
        "formula": "C5H13NO",
        "kegg_id": "C00114",
        "hmdb_id": "HMDB0000097",
        "class": "other",
    },
    "Carnitine": {
        "monoisotopic_mass": 161.10519,
        "formula": "C7H15NO3",
        "kegg_id": "C00318",
        "hmdb_id": "HMDB0000062",
        "class": "other",
    },
    "Acetylcarnitine": {
        "monoisotopic_mass": 203.11576,
        "formula": "C9H17NO4",
        "kegg_id": "C02571",
        "hmdb_id": "HMDB0000201",
        "class": "other",
    },
    "Betaine": {
        "monoisotopic_mass": 117.07898,
        "formula": "C5H11NO2",
        "kegg_id": "C00719",
        "hmdb_id": "HMDB0000043",
        "class": "other",
    },
    "Glutathione": {
        "monoisotopic_mass": 307.08381,
        "formula": "C10H17N3O6S",
        "kegg_id": "C00051",
        "hmdb_id": "HMDB0000125",
        "class": "other",
    },
    "Hypoxanthine": {
        "monoisotopic_mass": 136.03854,
        "formula": "C5H4N4O",
        "kegg_id": "C00262",
        "hmdb_id": "HMDB0000157",
        "class": "nucleotide",
    },
    "Xanthine": {
        "monoisotopic_mass": 152.03344,
        "formula": "C5H4N4O2",
        "kegg_id": "C00385",
        "hmdb_id": "HMDB0000292",
        "class": "nucleotide",
    },
    "Cholesterol": {
        "monoisotopic_mass": 386.35486,
        "formula": "C27H46O",
        "kegg_id": "C00187",
        "hmdb_id": "HMDB0000067",
        "class": "sterol",
    },
    "Cortisol": {
        "monoisotopic_mass": 362.20932,
        "formula": "C21H30O5",
        "kegg_id": "C00735",
        "hmdb_id": "HMDB0000063",
        "class": "sterol",
    },
    "Vitamin C": {
        "monoisotopic_mass": 176.03209,
        "formula": "C6H8O6",
        "kegg_id": "C00072",
        "hmdb_id": "HMDB0000044",
        "class": "vitamin",
    },
    "Nicotinamide": {
        "monoisotopic_mass": 122.04801,
        "formula": "C6H6N2O",
        "kegg_id": "C00153",
        "hmdb_id": "HMDB0001406",
        "class": "vitamin",
    },
    "Pantothenic acid": {
        "monoisotopic_mass": 219.11067,
        "formula": "C9H17NO5",
        "kegg_id": "C00864",
        "hmdb_id": "HMDB0000210",
        "class": "vitamin",
    },
}


# Common adducts with mass adjustments
COMMON_ADDUCTS: Dict[str, float] = {
    "[M+H]+": 1.007276,
    "[M+Na]+": 22.989218,
    "[M+K]+": 38.963158,
    "[M+NH4]+": 18.034164,
    "[M-H2O+H]+": -17.002740,
    "[M-H]-": -1.007276,
    "[M+Cl]-": 34.969402,
    "[M+FA-H]-": 44.998201,
    "[M-H2O-H]-": -19.018390,
    "[M+CH3COO]-": 59.013851,
}


# Lipid m/z range classification (rough estimates for MSI level 3)
LIPID_MZ_RANGES: Dict[str, Tuple[float, float]] = {
    "fatty_acid": (200.0, 400.0),
    "lysophospholipid": (400.0, 600.0),
    "phospholipid": (600.0, 900.0),
    "sphingolipid": (500.0, 850.0),
    "glycerolipid_TG": (800.0, 1000.0),
}


class MetabolomicsAnnotationError(Exception):
    """Base exception for metabolomics annotation operations."""

    pass


class MetabolomicsAnnotationService:
    """
    Stateless annotation service for metabolomics data.

    Provides m/z-based metabolite identification against a bundled reference
    database and lipid class classification based on annotations or m/z ranges.
    """

    def __init__(self):
        """Initialize the metabolomics annotation service (stateless)."""
        logger.debug("Initializing stateless MetabolomicsAnnotationService")

    def _create_ir_annotate_by_mz(
        self,
        mz_column: str,
        ppm_tolerance: float,
        adducts: Optional[List[str]],
        ion_mode: str,
    ) -> AnalysisStep:
        """Create IR for m/z annotation."""
        return AnalysisStep(
            operation="metabolomics.annotation.annotate_by_mz",
            tool_name="annotate_metabolites",
            description="Annotate metabolites by m/z matching against reference database",
            library="numpy",
            code_template="""# Metabolite annotation by m/z
from lobster.services.annotation.metabolomics_annotation_service import MetabolomicsAnnotationService

service = MetabolomicsAnnotationService()
adata_ann, stats, _ = service.annotate_by_mz(
    adata,
    mz_column={{ mz_column | tojson }},
    ppm_tolerance={{ ppm_tolerance }},
    ion_mode={{ ion_mode | tojson }}
)
print(f"Annotated: {stats['n_annotated']}/{stats['n_annotated'] + stats['n_unannotated']} ({stats['annotation_rate_pct']:.1f}%)")""",
            imports=[
                "from lobster.services.annotation.metabolomics_annotation_service import MetabolomicsAnnotationService"
            ],
            parameters={
                "mz_column": mz_column,
                "ppm_tolerance": ppm_tolerance,
                "adducts": adducts,
                "ion_mode": ion_mode,
            },
            parameter_schema={
                "mz_column": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="mz",
                    required=False,
                    description="Column in var containing m/z values",
                ),
                "ppm_tolerance": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=10.0,
                    required=False,
                    validation_rule="ppm_tolerance > 0",
                    description="Mass accuracy tolerance in ppm",
                ),
                "ion_mode": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="positive",
                    required=False,
                    validation_rule="ion_mode in ['positive', 'negative']",
                    description="Ionization mode for adduct selection",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_annotated"],
        )

    def _create_ir_classify_lipids(
        self,
        annotation_column: str,
    ) -> AnalysisStep:
        """Create IR for lipid classification."""
        return AnalysisStep(
            operation="metabolomics.annotation.classify_lipids",
            tool_name="analyze_lipid_classes",
            description="Classify features by lipid class from annotations or m/z ranges",
            library="numpy",
            code_template="""# Lipid class classification
from lobster.services.annotation.metabolomics_annotation_service import MetabolomicsAnnotationService

service = MetabolomicsAnnotationService()
adata_lipid, stats, _ = service.classify_lipids(
    adata,
    annotation_column={{ annotation_column | tojson }}
)
print(f"Classified into {stats['n_classes']} lipid classes")""",
            imports=[
                "from lobster.services.annotation.metabolomics_annotation_service import MetabolomicsAnnotationService"
            ],
            parameters={"annotation_column": annotation_column},
            parameter_schema={
                "annotation_column": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="annotation_class",
                    required=False,
                    description="Column in var with annotation class labels",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_lipid"],
        )

    def annotate_by_mz(
        self,
        adata: anndata.AnnData,
        mz_column: str = "mz",
        ppm_tolerance: float = 10.0,
        adducts: Optional[List[str]] = None,
        ion_mode: str = "positive",
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Annotate metabolites by m/z matching against bundled reference database.

        For each feature's m/z, computes candidate neutral masses for each adduct,
        then matches against METABOLITE_REFERENCE_DB within ppm tolerance.
        Assigns MSI level 2 (putative annotation) for m/z-only matches.

        Args:
            adata: AnnData with m/z values in var
            mz_column: Column in var containing observed m/z values
            ppm_tolerance: Tolerance in ppm for mass matching
            adducts: List of adduct types to consider (default: mode-appropriate set)
            ion_mode: "positive" or "negative" for adduct selection

        Returns:
            Tuple of (AnnData with annotations in var, stats dict, AnalysisStep)
        """
        try:
            logger.info(
                f"Starting m/z annotation: ppm={ppm_tolerance}, mode={ion_mode}"
            )
            adata_ann = adata.copy()

            if mz_column not in adata_ann.var.columns:
                raise MetabolomicsAnnotationError(
                    f"m/z column '{mz_column}' not found in var. "
                    f"Available: {list(adata_ann.var.columns)}"
                )

            mz_values = adata_ann.var[mz_column].values.astype(float)

            # Select adducts based on ion mode
            if adducts is None:
                if ion_mode == "positive":
                    adducts = ["[M+H]+", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M-H2O+H]+"]
                else:
                    adducts = [
                        "[M-H]-",
                        "[M+Cl]-",
                        "[M+FA-H]-",
                        "[M-H2O-H]-",
                        "[M+CH3COO]-",
                    ]

            # Initialize annotation arrays
            n_features = len(mz_values)
            ann_names = [""] * n_features
            ann_hmdb = [""] * n_features
            ann_kegg = [""] * n_features
            ann_class = [""] * n_features
            ann_msi_level = [4] * n_features  # Default: unknown (MSI level 4)
            ann_ppm_error = [np.nan] * n_features
            ann_adduct = [""] * n_features

            n_annotated = 0

            for i, obs_mz in enumerate(mz_values):
                if np.isnan(obs_mz) or obs_mz <= 0:
                    continue

                best_match = None
                best_ppm = float("inf")
                best_adduct_name = ""

                for adduct_name in adducts:
                    if adduct_name not in COMMON_ADDUCTS:
                        continue
                    adduct_mass = COMMON_ADDUCTS[adduct_name]

                    # Calculate neutral mass from observed m/z
                    if ion_mode == "positive":
                        neutral_mass = obs_mz - adduct_mass
                    else:
                        neutral_mass = obs_mz + abs(adduct_mass)

                    if neutral_mass <= 0:
                        continue

                    # Match against reference database
                    for name, ref in METABOLITE_REFERENCE_DB.items():
                        theoretical_mass = ref["monoisotopic_mass"]
                        ppm_error = (
                            abs(neutral_mass - theoretical_mass)
                            / theoretical_mass
                            * 1e6
                        )
                        if ppm_error <= ppm_tolerance and ppm_error < best_ppm:
                            best_ppm = ppm_error
                            best_match = (name, ref)
                            best_adduct_name = adduct_name

                if best_match is not None:
                    name, ref = best_match
                    ann_names[i] = name
                    ann_hmdb[i] = ref.get("hmdb_id", "")
                    ann_kegg[i] = ref.get("kegg_id", "")
                    ann_class[i] = ref.get("class", "")
                    ann_msi_level[i] = 2  # MSI level 2: putative annotation
                    ann_ppm_error[i] = best_ppm
                    ann_adduct[i] = best_adduct_name
                    n_annotated += 1

            # Store annotations in var
            adata_ann.var["annotation_name"] = ann_names
            adata_ann.var["annotation_hmdb"] = ann_hmdb
            adata_ann.var["annotation_kegg"] = ann_kegg
            adata_ann.var["annotation_class"] = ann_class
            adata_ann.var["annotation_msi_level"] = ann_msi_level
            adata_ann.var["annotation_ppm_error"] = ann_ppm_error
            adata_ann.var["annotation_adduct"] = ann_adduct

            n_unannotated = n_features - n_annotated
            rate = (n_annotated / n_features * 100) if n_features > 0 else 0.0

            # MSI level distribution
            msi_dist = {}
            for level in ann_msi_level:
                msi_dist[f"MSI_{level}"] = msi_dist.get(f"MSI_{level}", 0) + 1

            stats = {
                "n_annotated": n_annotated,
                "n_unannotated": n_unannotated,
                "annotation_rate_pct": float(rate),
                "msi_level_distribution": msi_dist,
                "ppm_tolerance": ppm_tolerance,
                "ion_mode": ion_mode,
                "adducts_used": adducts,
                "reference_db_size": len(METABOLITE_REFERENCE_DB),
                "analysis_type": "metabolomics_mz_annotation",
            }

            logger.info(
                f"Annotation complete: {n_annotated}/{n_features} annotated ({rate:.1f}%)"
            )

            ir = self._create_ir_annotate_by_mz(
                mz_column, ppm_tolerance, adducts, ion_mode
            )
            return adata_ann, stats, ir

        except Exception as e:
            logger.exception(f"Error in m/z annotation: {e}")
            raise MetabolomicsAnnotationError(f"m/z annotation failed: {str(e)}")

    def classify_lipids(
        self,
        adata: anndata.AnnData,
        annotation_column: str = "annotation_class",
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Classify features by lipid class based on annotations or m/z ranges.

        For annotated features: groups by annotation_class column.
        For unannotated features: uses m/z ranges for rough classification (MSI level 3).

        Args:
            adata: AnnData object (ideally after annotate_by_mz)
            annotation_column: Column in var with class labels from annotation

        Returns:
            Tuple of (AnnData with lipid_class in var, stats dict, AnalysisStep)
        """
        try:
            logger.info("Starting lipid class classification")
            adata_lipid = adata.copy()

            if hasattr(adata_lipid.X, "toarray"):
                X = adata_lipid.X.toarray().astype(np.float64)
            else:
                X = np.array(adata_lipid.X, dtype=np.float64)

            n_features = adata_lipid.n_vars
            lipid_classes = ["unclassified"] * n_features
            msi_from_range = False

            # Try annotation-based classification first
            has_annotations = (
                annotation_column in adata_lipid.var.columns
                and adata_lipid.var[annotation_column]
                .astype(str)
                .str.strip()
                .ne("")
                .any()
            )

            if has_annotations:
                for i in range(n_features):
                    cls = str(adata_lipid.var[annotation_column].iloc[i]).strip()
                    if cls and cls != "" and cls != "nan":
                        lipid_classes[i] = cls
            else:
                logger.info(
                    "No annotations found; using m/z range classification (MSI level 3)"
                )
                msi_from_range = True

            # For unclassified features, attempt m/z range classification
            if "mz" in adata_lipid.var.columns:
                mz_values = adata_lipid.var["mz"].values.astype(float)
                for i in range(n_features):
                    if lipid_classes[i] == "unclassified" and not np.isnan(
                        mz_values[i]
                    ):
                        mz = mz_values[i]
                        for class_name, (low, high) in LIPID_MZ_RANGES.items():
                            if low <= mz <= high:
                                lipid_classes[i] = class_name
                                break

            adata_lipid.var["lipid_class"] = lipid_classes

            # Compute per-class statistics
            class_counts: Dict[str, int] = {}
            class_intensity: Dict[str, Dict[str, float]] = {}
            for i in range(n_features):
                cls = lipid_classes[i]
                class_counts[cls] = class_counts.get(cls, 0) + 1
                feature_intensities = X[:, i]
                valid = feature_intensities[~np.isnan(feature_intensities)]
                if cls not in class_intensity:
                    class_intensity[cls] = {"total": 0.0, "mean": 0.0, "count": 0}
                if len(valid) > 0:
                    class_intensity[cls]["total"] += float(np.sum(valid))
                    class_intensity[cls]["mean"] += float(np.mean(valid))
                    class_intensity[cls]["count"] += 1

            # Average the mean intensities
            for cls in class_intensity:
                cnt = class_intensity[cls]["count"]
                if cnt > 0:
                    class_intensity[cls]["mean"] /= cnt

            n_classes = len([c for c in class_counts if c != "unclassified"])

            stats = {
                "n_classes": n_classes,
                "class_counts": class_counts,
                "class_intensity_summary": class_intensity,
                "msi_level": 3 if msi_from_range else 2,
                "analysis_type": "metabolomics_lipid_classification",
            }

            logger.info(
                f"Lipid classification complete: {n_classes} classes identified"
            )

            ir = self._create_ir_classify_lipids(annotation_column)
            return adata_lipid, stats, ir

        except Exception as e:
            logger.exception(f"Error in lipid classification: {e}")
            raise MetabolomicsAnnotationError(f"Lipid classification failed: {str(e)}")
