"""
Peptide analysis configuration — amino acid property tables, enzyme patterns, thresholds.
"""

# Standard amino acid one-letter codes
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY_KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Amino acid molecular weights (monoisotopic, Da)
AA_MOLECULAR_WEIGHT = {
    "A": 71.03711, "R": 156.10111, "N": 114.04293, "D": 115.02694,
    "C": 103.00919, "E": 129.04259, "Q": 128.05858, "G": 57.02146,
    "H": 137.05891, "I": 113.08406, "L": 113.08406, "K": 128.09496,
    "M": 131.04049, "F": 147.06841, "P": 97.05276, "S": 87.03203,
    "T": 101.04768, "W": 186.07931, "Y": 163.06333, "V": 99.06841,
}

# Amino acid charge at pH 7.4
AA_CHARGE = {
    "R": 1.0, "K": 1.0, "H": 0.1, "D": -1.0, "E": -1.0,
}

# Activity prediction thresholds
AMP_THRESHOLDS = {
    "charge_range": (2, 9),
    "hydro_ratio_range": (0.4, 0.6),
    "boman_max": 2.48,
    "length_range": (10, 50),
    "hm_min": 0.5,
}

CPP_THRESHOLDS = {
    "arg_ratio_min": 0.2,
    "charge_min": 3.0,
    "length_range": (5, 30),
}

TOXICITY_THRESHOLDS = {
    "hydro_ratio_min": 0.6,
    "length_max": 30,
    "charge_min": 5.0,
    "trp_ratio_min": 0.1,
}

# Default property columns stored in AnnData.obs
DEFAULT_PROPERTY_COLUMNS = [
    "peptide_molecular_weight",
    "peptide_isoelectric_point",
    "peptide_charge",
    "peptide_gravy",
    "peptide_boman_index",
    "peptide_aliphatic_index",
    "peptide_instability_index",
    "peptide_hydrophobic_moment",
    "peptide_hydrophobic_ratio",
    "peptide_length",
]
