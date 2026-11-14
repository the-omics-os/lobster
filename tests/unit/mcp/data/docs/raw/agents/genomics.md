# Genomics



import { AgentHero } from '@/components/AgentHero';
import { NextSteps } from '@/components/NextSteps';
import { Rocket, GraduationCap, Database } from 'lucide-react';

<AgentHero
  name="lobster-genomics"
  tier="free"
  problem="Variant analysis: VCF parsing, annotation, GWAS, and population genetics"
  inputs={["VCF", "BCF", "gVCF", "Phenotype TSV"]}
  outputs={["Annotated Variants", "Manhattan Plots", "Allele Frequencies", "GWAS Results"]}
  install="pip install lobster-genomics"
  difficulty="intermediate"
  agents={[
  { name: "genomics_expert", role: "Variant analysis and GWAS" }
]}
/>

Agents [#agents]

genomics_expert [#genomics_expert]

Specialized agent for genomic variant analysis and GWAS.

**Capabilities:**

* VCF file loading and parsing
* Variant annotation (dbSNP, ClinVar)
* GWAS analysis
* Population genetics statistics
* Variant filtering and QC
* Ensembl VEP variant consequence prediction (SIFT, PolyPhen scores)
* Sequence retrieval (genomic, cDNA, CDS, protein) via Ensembl REST API
* Cross-database ID mapping (Ensembl, UniProt, HGNC)

Example Workflows [#example-workflows]

Variant Annotation [#variant-annotation]

```text
User: Analyze the variants in my VCF file and annotate with ClinVar

[genomics_expert]
- Loads VCF with cyvcf2
- Parses variant records
- Queries ClinVar annotations
- Filters by clinical significance
- Reports pathogenic/likely pathogenic variants
```

GWAS Analysis [#gwas-analysis]

```text
User: Run a GWAS analysis for the phenotype in my sample metadata

[genomics_expert]
- Loads genotype data from VCF
- Associates with phenotype data
- Calculates association statistics
- Generates Manhattan plot
- Reports genome-wide significant loci
```

Population Genetics [#population-genetics]

```text
User: Calculate allele frequencies across populations

[genomics_expert]
- Groups samples by population annotation
- Calculates allele frequencies per population
- Computes Fst between populations
- Identifies population-specific variants
```

Variant Consequence Prediction [#variant-consequence-prediction]

```text
User: What is the predicted consequence of the variant rs121913529
      (TP53 R175H)?

[genomics_expert]
- Queries Ensembl VEP REST API with variant identifier
- Returns predicted consequences:
  missense_variant (impact: HIGH)
  SIFT: deleterious (score: 0.0)
  PolyPhen: probably_damaging (score: 0.999)
- Reports affected transcript and protein position
```

Sequence Retrieval [#sequence-retrieval]

```text
User: Get the protein sequence for BRCA1

[genomics_expert]
- Queries Ensembl REST API for BRCA1 gene
- Retrieves protein sequence in FASTA format
- Returns: 1863 amino acid sequence (ENSP00000350283)
- Also available: genomic DNA, cDNA, CDS sequences
```

Dependencies [#dependencies]

| Library      | Purpose                                       |
| ------------ | --------------------------------------------- |
| **cyvcf2**   | Fast VCF parsing                              |
| **pyranges** | Genomic interval operations                   |
| **numpy**    | Numerical computations                        |
| **pandas**   | Data manipulation                             |
| **requests** | HTTP client for Ensembl and UniProt REST APIs |

Services [#services]

lobster-genomics includes domain-specific services bundled with the package:

| Service                      | Purpose                                                                     |
| ---------------------------- | --------------------------------------------------------------------------- |
| **VariantAnnotationService** | VCF variant annotation (ClinVar, dbSNP)                                     |
| **GWASService**              | Genome-wide association studies                                             |
| **GenomicsQualityService**   | Variant quality control and filtering                                       |
| **EnsemblService**           | Gene lookup, VEP variant consequences, sequence retrieval, cross-references |
| **UniProtService**           | Protein information, search, and ID mapping                                 |

Services are installed automatically with the agent package.

Configuration [#configuration]

```toml
# .lobster_workspace/config.toml
enabled = ["genomics_expert"]
```

Access [#access]

lobster-genomics is free and open source. Install and use without any license or activation.

VCF Support [#vcf-support]

genomics\_expert handles various VCF formats:

| Format               | Support      |
| -------------------- | ------------ |
| VCF 4.0+             | Full support |
| gVCF                 | Full support |
| BCF (binary VCF)     | Full support |
| Compressed (.vcf.gz) | Full support |

<NextSteps
  items={[
{
  href: "/docs/getting-started",
  title: "Getting Started",
  description: "Quick setup guide for genomics analysis",
  icon: <Rocket />
},
{
  href: "/docs/agents/ml",
  title: "Machine Learning",
  description: "GWAS-derived feature selection and analysis",
  icon: <GraduationCap />
},
{
  href: "/docs/guides/data-formats",
  title: "Data Loading Guide",
  description: "Loading VCF and variant data",
  icon: <Database />
}
]}
/>
