"""Coarse lineage groups for the 45 provided cell-type labels.

This grouping is ours, made for visualization only (a plot cannot show 45
colors). It is not used as a model input or for any metric.
"""

LINEAGE = {
    # T cells
    "CD4+ T activated": "T", "CD4+ T naive": "T", "CD4+ T activated integrinB7+": "T",
    "CD4+ T CD314+ CD45RA+": "T", "T reg": "T",
    "CD8+ T naive": "T", "CD8+ T CD57+ CD45RO+": "T", "CD8+ T CD57+ CD45RA+": "T",
    "CD8+ T TIGIT+ CD45RO+": "T", "CD8+ T TIGIT+ CD45RA+": "T", "CD8+ T CD49f+": "T",
    "CD8+ T CD69+ CD45RO+": "T", "CD8+ T CD69+ CD45RA+": "T",
    "CD8+ T naive CD127+ CD26- CD101-": "T",
    "MAIT": "T", "gdT CD158b+": "T", "gdT TCRVD2+": "T", "dnT": "T", "T prog cycling": "T",
    # NK and innate lymphoid cells
    "NK": "NK/ILC", "NK CD158e1+": "NK/ILC", "ILC1": "NK/ILC", "ILC": "NK/ILC",
    # B cells
    "Naive CD20+ B IGKC+": "B", "Naive CD20+ B IGKC-": "B", "Transitional B": "B",
    "B1 B IGKC+": "B", "B1 B IGKC-": "B",
    # Antibody-secreting cells
    "Plasma cell IGKC+": "Plasma", "Plasma cell IGKC-": "Plasma",
    "Plasmablast IGKC+": "Plasma", "Plasmablast IGKC-": "Plasma",
    # Monocytes and dendritic cells
    "CD14+ Mono": "Myeloid", "CD16+ Mono": "Myeloid", "cDC1": "Myeloid", "cDC2": "Myeloid",
    "pDC": "Myeloid",
    # Erythroid
    "Proerythroblast": "Erythroid", "Erythroblast": "Erythroid", "Normoblast": "Erythroid",
    "Reticulocyte": "Erythroid",
    # Progenitors
    "HSC": "Progenitor", "G/M prog": "Progenitor", "Lymph prog": "Progenitor",
    "MK/E prog": "Progenitor",
}

LINEAGE_ORDER = ["T", "NK/ILC", "B", "Plasma", "Myeloid", "Erythroid", "Progenitor"]


def to_lineage(cell_types):
    """Map cell-type labels to lineage groups; raises if a label is unknown."""
    missing = sorted(set(cell_types) - set(LINEAGE))
    if missing:
        raise KeyError(f"cell types without a lineage: {missing}")
    return [LINEAGE[c] for c in cell_types]
