import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

samples = data.get("data", {}).get("samples", [])
print(f"Total samples: {len(samples)}")
print(f"Sample count field: {data.get('data', {}).get('sample_count', 0)}")

if samples:
    s = samples[0]
    print(f"\nFirst sample keys: {list(s.keys())}")
    print(f"organism: {s.get('organism', 'N/A')}")
    print(f"library_strategy: {s.get('library_strategy', 'N/A')}")
    print(f"sample_type: {s.get('sample_type', 'N/A')}")
    print(f"host: {s.get('host', 'N/A')}")
    print(f"disease: {s.get('disease', 'N/A')}")
    print(f"body_site: {s.get('body_site', 'N/A')}")
    print(f"isolation_source: {s.get('isolation_source', 'N/A')}")

    # Count organisms
    organisms = {}
    hosts = {}
    strategies = {}
    for sample in samples[:10]:  # First 10
        org = sample.get('organism', 'Unknown')
        organisms[org] = organisms.get(org, 0) + 1
        host = sample.get('host', 'Unknown')
        hosts[host] = hosts.get(host, 0) + 1
        strat = sample.get('library_strategy', 'Unknown')
        strategies[strat] = strategies.get(strat, 0) + 1

    print(f"\nOrganisms (first 10): {organisms}")
    print(f"Hosts (first 10): {hosts}")
    print(f"Library strategies (first 10): {strategies}")
