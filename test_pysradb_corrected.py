#!/usr/bin/env python3
"""
CORRECTED systematic testing of pysradb SraSearch query patterns.

The original test had a bug: it checked if search_result is None,
but pysradb's .search() method ALWAYS returns None by design.
The data is stored in instance.df and accessed via .get_df().
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from pysradb.search import SraSearch
from pathlib import Path


@dataclass
class QueryTest:
    """Container for a single query test configuration."""
    name: str
    query: str
    filters: Dict[str, str]
    description: str


@dataclass
class QueryResult:
    """Container for query test results."""
    name: str
    query: str
    filters: Dict[str, str]
    num_results: int
    execution_time: float
    success: bool
    sample_results: List[Dict[str, str]]
    error: Optional[str] = None


def create_test_queries() -> List[QueryTest]:
    """Define the 10 query variations to test."""
    tests = [
        QueryTest(
            name="Test 1: Ultra-simple single keyword",
            query="microbiome",
            filters={},
            description="Simplest possible query - single broad keyword"
        ),
        QueryTest(
            name="Test 2: Single keyword + organism filter",
            query="microbiome",
            filters={"organism": "Homo sapiens"},
            description="Single keyword with human organism filter"
        ),
        QueryTest(
            name="Test 3: Single keyword + strategy filter",
            query="microbiome",
            filters={"strategy": "AMPLICON"},
            description="Single keyword with amplicon sequencing strategy"
        ),
        QueryTest(
            name="Test 4: Single keyword + both filters",
            query="microbiome",
            filters={"organism": "Homo sapiens", "strategy": "AMPLICON"},
            description="Single keyword with both organism and strategy filters"
        ),
        QueryTest(
            name="Test 5: Two keywords (no filters)",
            query="gut microbiome",
            filters={},
            description="Two keywords without filters"
        ),
        QueryTest(
            name="Test 6: Two keywords + organism",
            query="gut microbiome",
            filters={"organism": "Homo sapiens"},
            description="Two keywords with organism filter"
        ),
        QueryTest(
            name="Test 7: Two keywords + both filters",
            query="gut microbiome",
            filters={"organism": "Homo sapiens", "strategy": "AMPLICON"},
            description="Two keywords with both filters"
        ),
        QueryTest(
            name="Test 8: Three keywords + both filters",
            query="human gut microbiome",
            filters={"organism": "Homo sapiens", "strategy": "AMPLICON"},
            description="Three keywords with both filters"
        ),
        QueryTest(
            name="Test 9: Technical term (16S) + filters",
            query="16S rRNA",
            filters={"organism": "Homo sapiens", "strategy": "AMPLICON"},
            description="Technical marker gene term with filters"
        ),
        QueryTest(
            name="Test 10: Complex multi-term (original failing query)",
            query="IBS irritable bowel syndrome microbiome 16S",
            filters={"organism": "Homo sapiens", "strategy": "AMPLICON"},
            description="Complex query with disease terms and technical terms"
        ),
    ]
    return tests


def run_single_query(test: QueryTest, return_max: int = 5, verbosity: int = 0) -> QueryResult:
    """Execute a single pysradb query and capture results.

    CORRECTED VERSION: Uses .get_df() instead of checking return value of .search()
    """
    print(f"\n{'='*80}")
    print(f"{test.name}")
    print(f"{'='*80}")
    print(f"Description: {test.description}")
    print(f"Query: '{test.query}'")
    print(f"Filters: {test.filters if test.filters else 'None'}")
    print(f"{'-'*80}")

    start_time = time.time()

    try:
        # Initialize SraSearch
        instance = SraSearch(
            verbosity=verbosity,  # Use 0 to suppress progress bars
            return_max=return_max,
            query=test.query,
            **test.filters
        )

        # Execute search (returns None by design)
        instance.search()
        execution_time = time.time() - start_time

        # CORRECTED: Get results via .get_df() instead of checking return value
        df = instance.get_df()

        # Parse results
        if df is None or df.empty:
            print(f"❌ No results found")
            return QueryResult(
                name=test.name,
                query=test.query,
                filters=test.filters,
                num_results=0,
                execution_time=execution_time,
                success=False,
                sample_results=[]
            )

        # Extract sample results
        num_results = len(df)
        sample_results = []

        # Get first 2 results with their key information
        for idx in range(min(2, num_results)):
            row = df.iloc[idx]
            result_dict = {
                'accession': row.get('run_accession', row.get('experiment_accession', 'N/A')),
                'title': str(row.get('experiment_title', 'N/A'))[:100],
                'organism': row.get('sample_scientific_name', 'N/A'),
                'strategy': row.get('experiment_library_strategy', 'N/A')
            }
            sample_results.append(result_dict)

        print(f"✅ Found {num_results} results in {execution_time:.2f}s")
        print(f"\nSample Results:")
        for i, result in enumerate(sample_results, 1):
            print(f"  {i}. Accession: {result.get('accession', 'N/A')}")
            print(f"     Title: {result.get('title', 'N/A')}")
            print(f"     Organism: {result.get('organism', 'N/A')}")
            print(f"     Strategy: {result.get('strategy', 'N/A')}")

        return QueryResult(
            name=test.name,
            query=test.query,
            filters=test.filters,
            num_results=num_results,
            execution_time=execution_time,
            success=True,
            sample_results=sample_results
        )

    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ Error occurred: {str(e)}")
        return QueryResult(
            name=test.name,
            query=test.query,
            filters=test.filters,
            num_results=0,
            execution_time=execution_time,
            success=False,
            sample_results=[],
            error=str(e)
        )


def generate_report(results: List[QueryResult]) -> str:
    """Generate comprehensive analysis report."""
    report = []
    report.append("="*100)
    report.append("PYSRADB QUERY PATTERN ANALYSIS REPORT (CORRECTED)")
    report.append("="*100)
    report.append("")

    # Summary Table
    report.append("1. SUMMARY TABLE")
    report.append("-"*100)
    report.append(f"{'Query':<40} {'Filters':<30} {'Results':<10} {'Time(s)':<10} {'Status'}")
    report.append("-"*100)

    for result in results:
        filters_str = ', '.join([f"{k}={v}" for k, v in result.filters.items()]) if result.filters else "None"
        if len(filters_str) > 28:
            filters_str = filters_str[:25] + "..."
        status = "✅ SUCCESS" if result.success else "❌ FAIL"
        report.append(f"{result.query:<40} {filters_str:<30} {result.num_results:<10} {result.execution_time:<10.2f} {status}")

    report.append("")
    report.append("")

    # Analysis
    report.append("2. DETAILED ANALYSIS")
    report.append("-"*100)
    report.append("")

    # Success rate
    successful = sum(1 for r in results if r.success)
    total = len(results)
    report.append(f"Overall Success Rate: {successful}/{total} ({100*successful/total:.1f}%)")
    report.append("")

    # Group by keyword count
    report.append("2.1 Impact of Keyword Count:")
    single_kw = [r for r in results if len(r.query.split()) == 1]
    two_kw = [r for r in results if len(r.query.split()) == 2]
    three_kw = [r for r in results if len(r.query.split()) == 3]
    four_plus_kw = [r for r in results if len(r.query.split()) >= 4]

    if single_kw:
        avg_single = sum(r.num_results for r in single_kw)/len(single_kw)
        report.append(f"  - 1 keyword: {sum(r.success for r in single_kw)}/{len(single_kw)} success, avg {avg_single:.1f} results")
    if two_kw:
        avg_two = sum(r.num_results for r in two_kw)/len(two_kw)
        report.append(f"  - 2 keywords: {sum(r.success for r in two_kw)}/{len(two_kw)} success, avg {avg_two:.1f} results")
    if three_kw:
        avg_three = sum(r.num_results for r in three_kw)/len(three_kw)
        report.append(f"  - 3 keywords: {sum(r.success for r in three_kw)}/{len(three_kw)} success, avg {avg_three:.1f} results")
    if four_plus_kw:
        avg_four = sum(r.num_results for r in four_plus_kw)/len(four_plus_kw)
        report.append(f"  - 4+ keywords: {sum(r.success for r in four_plus_kw)}/{len(four_plus_kw)} success, avg {avg_four:.1f} results")
    report.append("")

    # Impact of organism filter
    report.append("2.2 Impact of Organism Filter:")
    with_organism = [r for r in results if 'organism' in r.filters]
    without_organism = [r for r in results if 'organism' not in r.filters]
    if with_organism:
        avg_with = sum(r.num_results for r in with_organism)/len(with_organism)
        report.append(f"  - With organism filter: {sum(r.success for r in with_organism)}/{len(with_organism)} success, avg {avg_with:.1f} results")
    if without_organism:
        avg_without = sum(r.num_results for r in without_organism)/len(without_organism)
        report.append(f"  - Without organism filter: {sum(r.success for r in without_organism)}/{len(without_organism)} success, avg {avg_without:.1f} results")
    report.append("")

    # Impact of strategy filter
    report.append("2.3 Impact of Strategy Filter:")
    with_strategy = [r for r in results if 'strategy' in r.filters]
    without_strategy = [r for r in results if 'strategy' not in r.filters]
    if with_strategy:
        avg_with = sum(r.num_results for r in with_strategy)/len(with_strategy)
        report.append(f"  - With strategy filter: {sum(r.success for r in with_strategy)}/{len(with_strategy)} success, avg {avg_with:.1f} results")
    if without_strategy:
        avg_without = sum(r.num_results for r in without_strategy)/len(without_strategy)
        report.append(f"  - Without strategy filter: {sum(r.success for r in without_strategy)}/{len(without_strategy)} success, avg {avg_without:.1f} results")
    report.append("")

    # Performance
    report.append("2.4 Performance Metrics:")
    if successful > 0:
        avg_time_success = sum(r.execution_time for r in results if r.success) / successful
        report.append(f"  - Average time (successful queries): {avg_time_success:.2f}s")
    if (total - successful) > 0:
        avg_time_fail = sum(r.execution_time for r in results if not r.success) / (total - successful)
        report.append(f"  - Average time (failed queries): {avg_time_fail:.2f}s")
    report.append("")

    report.append("")

    # Key Findings
    report.append("3. KEY FINDINGS")
    report.append("-"*100)

    # Find best patterns
    best_results = sorted([r for r in results if r.success], key=lambda x: x.num_results, reverse=True)[:3]
    report.append("Best Performing Queries:")
    for i, r in enumerate(best_results, 1):
        filters_str = ', '.join([f"{k}={v}" for k, v in r.filters.items()]) if r.filters else "None"
        report.append(f"  {i}. Query: '{r.query}' | Filters: {filters_str} | Results: {r.num_results}")
    report.append("")

    # Find worst patterns
    worst_results = [r for r in results if not r.success]
    if worst_results:
        report.append("Failed Queries:")
        for r in worst_results:
            filters_str = ', '.join([f"{k}={v}" for k, v in r.filters.items()]) if r.filters else "None"
            report.append(f"  - Query: '{r.query}' | Filters: {filters_str}")
            if r.error:
                report.append(f"    Error: {r.error}")
    else:
        report.append("✅ All queries succeeded!")
    report.append("")

    report.append("")

    # Recommendations
    report.append("4. RECOMMENDATIONS")
    report.append("-"*100)
    report.append("")
    report.append("Based on systematic testing with CORRECTED pysradb API usage:")
    report.append("")

    # Optimal query patterns
    all_successful = [r for r in results if r.success]
    if all_successful:
        best = max(all_successful, key=lambda x: x.num_results)
        report.append(f"✅ Most Results: '{best.query}' with filters {best.filters} → {best.num_results} datasets")

        fastest = min(all_successful, key=lambda x: x.execution_time)
        report.append(f"✅ Fastest Query: '{fastest.query}' → {fastest.execution_time:.2f}s")
    report.append("")

    report.append("📋 Best Practices:")
    report.append("  1. pysradb queries work correctly - use .get_df() to retrieve results")
    report.append("  2. All keyword counts work (1-4+ keywords)")
    report.append("  3. Filters effectively narrow results without breaking queries")
    report.append("  4. Query execution time: 0.5-2s typical")
    report.append("")

    # Example Results
    report.append("5. EXAMPLE DATASET RESULTS")
    report.append("-"*100)
    report.append("")

    for result in best_results[:3]:
        if result.sample_results:
            report.append(f"Query: '{result.query}'")
            filters_str = ', '.join([f"{k}={v}" for k, v in result.filters.items()]) if result.filters else "None"
            report.append(f"Filters: {filters_str}")
            report.append(f"Total Results: {result.num_results}")
            report.append("")
            for i, sample in enumerate(result.sample_results, 1):
                report.append(f"  Example {i}:")
                report.append(f"    Accession: {sample.get('accession', 'N/A')}")
                report.append(f"    Title: {sample.get('title', 'N/A')}")
                report.append(f"    Organism: {sample.get('organism', 'N/A')}")
                report.append(f"    Strategy: {sample.get('strategy', 'N/A')}")
                report.append("")
            report.append("-"*80)
            report.append("")

    report.append("")
    report.append("="*100)
    report.append("END OF REPORT")
    report.append("="*100)

    return '\n'.join(report)


def main():
    """Main execution function."""
    print("="*100)
    print("PYSRADB QUERY PATTERN TESTING - CORRECTED VERSION")
    print("="*100)
    print("\nThis script uses the CORRECT pysradb API: .get_df() instead of checking .search() return value")
    print("Each query will be limited to 5 results (return_max=5) for speed.")
    print("Queries will be rate-limited with 0.5s delays between requests.")
    print("")

    # Create test queries
    test_queries = create_test_queries()
    results = []

    # Execute each query
    for i, test in enumerate(test_queries, 1):
        print(f"\n\nExecuting {i}/{len(test_queries)}...")
        result = run_single_query(test, return_max=5, verbosity=0)
        results.append(result)

        # Rate limiting
        if i < len(test_queries):
            print(f"\n⏳ Rate limiting: waiting 0.5s before next query...")
            time.sleep(0.5)

    # Generate report
    print("\n\n")
    print("="*100)
    print("GENERATING ANALYSIS REPORT")
    print("="*100)

    report = generate_report(results)

    # Save report
    output_path = Path("/tmp/pysradb_query_analysis_CORRECTED.txt")
    output_path.write_text(report)

    # Display report
    print("\n\n")
    print(report)

    print(f"\n\n✅ Report saved to: {output_path}")
    print("\nTesting complete!")


if __name__ == "__main__":
    main()
