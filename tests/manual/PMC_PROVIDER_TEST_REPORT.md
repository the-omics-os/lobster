# PMC Provider Direct Testing Report

Test Date: 2025-11-10 18:08:03
Publications Tested: 5

## Executive Summary

- **PMC Available:** 5/5 publications (100%)
- **Successful Extractions:** 5/5 (100% of available)
- **Failed Extractions:** 0

## PMC-Available Papers

### ✅ [Nature] PMID:35042229

**PMC ID:** PMC8942855

**Extraction Time:** 3.51s
**Performance Status:** ⚠️ (SLOW)

**Quality Metrics:**
- Full text: 48,433 chars
- Methods section: 1,738 chars
- Results section: 1,738 chars
- Discussion section: 1,738 chars
- Tables extracted: 0
- Software tools detected: 6
- GitHub repos found: 0

**Software Tools:**
- GATK
- https://github.com/covid-19-Re
- https://github.com/dsfsi/covid19za
- https://github.com/hCoV-2019/pangolin
- https://github.com/laduplessis/bdskytools
- https://github.com/owid/covid-19-data

**Quality Issues:**
- ⚠️ No tables extracted

### ✅ [Cell Press] PMID:33861949

**PMC ID:** PMC9987169

**Extraction Time:** 2.25s
**Performance Status:** ⚠️ (SLOW)

**Quality Metrics:**
- Full text: 25,814 chars
- Methods section: 4,818 chars
- Results section: 4,001 chars
- Discussion section: 11,408 chars
- Tables extracted: 0
- Software tools detected: 2
- GitHub repos found: 0

**Software Tools:**
- R
- limma

**Quality Issues:**
- ⚠️ No tables extracted

### ✅ [Science] PMID:35324292

**PMC ID:** PMC12563655

**Extraction Time:** 2.78s
**Performance Status:** ⚠️ (SLOW)

**Quality Metrics:**
- Full text: 23,476 chars
- Methods section: 2,950 chars
- Results section: 2,950 chars
- Discussion section: 2,950 chars
- Tables extracted: 0
- Software tools detected: 0
- GitHub repos found: 0

**Quality Issues:**
- ⚠️ No tables extracted
- ⚠️ No software tools detected

### ✅ [PLOS] PMID:33534773

**PMC ID:** PMC7941810

**Extraction Time:** 2.36s
**Performance Status:** ⚠️ (SLOW)

**Quality Metrics:**
- Full text: 20 chars
- Methods section: 0 chars
- Results section: 0 chars
- Discussion section: 0 chars
- Tables extracted: 1
- Software tools detected: 0
- GitHub repos found: 0

**Quality Issues:**
- ⚠️ No methods section extracted
- ⚠️ No software tools detected

### ✅ [BMC] PMID:33388025

**PMC ID:** PMC12476222

**Extraction Time:** 2.61s
**Performance Status:** ⚠️ (SLOW)

**Quality Metrics:**
- Full text: 41,413 chars
- Methods section: 1,476 chars
- Results section: 1,476 chars
- Discussion section: 1,476 chars
- Tables extracted: 0
- Software tools detected: 1
- GitHub repos found: 0

**Software Tools:**
- r

**Quality Issues:**
- ⚠️ No tables extracted

## Non-PMC Papers

All tested papers are available in PMC.

## Performance Summary

- **Average extraction time:** 2.70s
- **Min extraction time:** 2.25s
- **Max extraction time:** 3.51s
- **Target compliance:** 0/5 extractions < 2.0s

### Performance by Publisher

| Publisher | Extraction Time | Status |
|-----------|----------------|--------|
| Nature | 3.51s | ⚠️ SLOW |
| Cell Press | 2.25s | ⚠️ SLOW |
| Science | 2.78s | ⚠️ SLOW |
| PLOS | 2.36s | ⚠️ SLOW |
| BMC | 2.61s | ⚠️ SLOW |

## Quality Summary

- **Total methods content:** 10,982 chars
- **Total tables extracted:** 1
- **Total software tools detected:** 9
- **Average methods per paper:** 2,196 chars
- **Average tables per paper:** 0.2
- **Average software tools per paper:** 1.8

## Recommendations

- ⚠️ **Slow extractions:** 5 papers exceeded 2.0s target - consider caching
