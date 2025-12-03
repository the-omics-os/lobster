#!/usr/bin/env python3
"""
ASM Test URLs for Statistical Validation

This file contains 10 diverse ASM journal articles across different journals
for comprehensive strategy testing. URLs selected to represent:
- Different ASM journals (JCM, mBio, AAC, AEM, IAI, etc.)
- Different publication years
- Different article types
- Different DOI formats

Used for statistical validation of ASM access strategies before production deployment.
"""

ASM_TEST_URLS = [
    {
        "url": "https://journals.asm.org/doi/10.1128/JCM.01893-20",
        "journal": "Journal of Clinical Microbiology (JCM)",
        "year": 2020,
        "doi": "10.1128/JCM.01893-20",
        "description": "COVID-19 diagnostic test",
        "article_type": "research article",
        "access_notes": "Most cited clinical diagnostic journal in ASM portfolio",
    },
    {
        "url": "https://journals.asm.org/doi/10.1128/mBio.02227-21",
        "journal": "mBio",
        "year": 2021,
        "doi": "10.1128/mBio.02227-21",
        "description": "Bacterial pathogenesis",
        "article_type": "research article",
        "access_notes": "mBio is fully open access (Gold OA)",
    },
    {
        "url": "https://journals.asm.org/doi/10.1128/AAC.01737-20",
        "journal": "Antimicrobial Agents and Chemotherapy (AAC)",
        "year": 2020,
        "doi": "10.1128/AAC.01737-20",
        "description": "Antibiotic resistance",
        "article_type": "research article",
        "access_notes": "Premier antimicrobial resistance journal",
    },
    {
        "url": "https://journals.asm.org/doi/10.1128/AEM.01234-21",
        "journal": "Applied and Environmental Microbiology (AEM)",
        "year": 2021,
        "doi": "10.1128/AEM.01234-21",
        "description": "Environmental microbiome",
        "article_type": "research article",
        "access_notes": "Environmental microbiology focus",
    },
    {
        "url": "https://journals.asm.org/doi/10.1128/IAI.00123-22",
        "journal": "Infection and Immunity (IAI)",
        "year": 2022,
        "doi": "10.1128/IAI.00123-22",
        "description": "Host-pathogen interactions",
        "article_type": "research article",
        "access_notes": "Immunology and host response focus",
    },
    {
        "url": "https://journals.asm.org/doi/10.1128/JVI.01456-21",
        "journal": "Journal of Virology (JVI)",
        "year": 2021,
        "doi": "10.1128/JVI.01456-21",
        "description": "Viral replication mechanisms",
        "article_type": "research article",
        "access_notes": "Leading virology journal in ASM portfolio",
    },
    {
        "url": "https://journals.asm.org/doi/10.1128/MMBR.00089-20",
        "journal": "Microbiology and Molecular Biology Reviews (MMBR)",
        "year": 2020,
        "doi": "10.1128/MMBR.00089-20",
        "description": "Review article on microbiome",
        "article_type": "review article",
        "access_notes": "Review journal - comprehensive articles",
    },
    {
        "url": "https://journals.asm.org/doi/10.1128/spectrum.01234-23",
        "journal": "Microbiology Spectrum",
        "year": 2023,
        "doi": "10.1128/spectrum.01234-23",
        "description": "Microbial genomics",
        "article_type": "research article",
        "access_notes": "Microbiology Spectrum is fully open access (Gold OA)",
    },
    {
        "url": "https://journals.asm.org/doi/10.1128/msystems.00567-22",
        "journal": "mSystems",
        "year": 2022,
        "doi": "10.1128/msystems.00567-22",
        "description": "Systems biology approach",
        "article_type": "research article",
        "access_notes": "mSystems is fully open access (Gold OA)",
    },
    {
        "url": "https://journals.asm.org/doi/10.1128/JB.00234-21",
        "journal": "Journal of Bacteriology (JB)",
        "year": 2021,
        "doi": "10.1128/JB.00234-21",
        "description": "Bacterial metabolism",
        "article_type": "research article",
        "access_notes": "Molecular bacteriology and genetics focus",
    },
]


def get_test_urls():
    """Return list of test URLs."""
    return [item["url"] for item in ASM_TEST_URLS]


def get_test_url_metadata():
    """Return full metadata for all test URLs."""
    return ASM_TEST_URLS


def get_journal_distribution():
    """Return count of articles per journal."""
    from collections import Counter

    return Counter(item["journal"] for item in ASM_TEST_URLS)


def get_article_type_distribution():
    """Return count of articles by type."""
    from collections import Counter

    return Counter(item["article_type"] for item in ASM_TEST_URLS)


def get_year_distribution():
    """Return count of articles by year."""
    from collections import Counter

    return Counter(item["year"] for item in ASM_TEST_URLS)


def get_open_access_count():
    """Return count of gold open access articles."""
    return sum(
        1
        for item in ASM_TEST_URLS
        if "open access" in item.get("access_notes", "").lower()
    )


if __name__ == "__main__":
    print("ASM Test URL Collection")
    print("=" * 80)
    print(f"Total URLs: {len(ASM_TEST_URLS)}")
    print(f"Unique journals: {len(get_journal_distribution())}")
    print(f"Gold Open Access: {get_open_access_count()}/{len(ASM_TEST_URLS)}")

    print(f"\nJournal Distribution:")
    for journal, count in get_journal_distribution().items():
        print(f"  - {journal}: {count}")

    print(f"\nArticle Types:")
    for article_type, count in get_article_type_distribution().items():
        print(f"  - {article_type}: {count}")

    print(f"\nYear Distribution:")
    for year, count in sorted(get_year_distribution().items()):
        print(f"  - {year}: {count}")

    print(f"\nAll URLs:")
    for i, item in enumerate(ASM_TEST_URLS, 1):
        print(f"  {i}. {item['journal']} ({item['year']}) - {item['article_type']}")
        print(f"     {item['url']}")
        print(f"     {item['access_notes']}")
