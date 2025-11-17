## UnifiedContentService Integration Testing Report

### Test Results:

**Nature** - PMID:35042229
- **Status**: ✅ Success
- **tier_used**: `full_pmc_xml`
- **source_type**: `pmc_xml`
- **extraction_time**: 2.52 seconds
- **methods_text_length**: 1738 chars
- **software_detected**: ['GATK', 'https://github.com/covid-19-Re', 'https://github.com/dsfsi/covid19za', 'https://github.com/hCoV-2019/pangolin', 'https://github.com/laduplessis/bdskytools', 'https://github.com/owid/covid-19-data']
- **tables_count**: 0
- **PMC-first strategy**: ✅ Working correctly

**Cell Press** - PMID:33861949
- **Status**: ✅ Success
- **tier_used**: `full_pmc_xml`
- **source_type**: `pmc_xml`
- **extraction_time**: 2.24 seconds
- **methods_text_length**: 4818 chars
- **software_detected**: ['R', 'limma']
- **tables_count**: 0
- **PMC-first strategy**: ✅ Working correctly

**Science** - PMID:35324292
- **Status**: ✅ Success
- **tier_used**: `full_pmc_xml`
- **source_type**: `pmc_xml`
- **extraction_time**: 2.98 seconds
- **methods_text_length**: 2950 chars
- **software_detected**: []
- **tables_count**: 0
- **PMC-first strategy**: ✅ Working correctly

**PLOS** - PMID:33534773
- **Status**: ✅ Success
- **tier_used**: `full_pmc_xml`
- **source_type**: `pmc_xml`
- **extraction_time**: 2.47 seconds
- **methods_text_length**: 0 chars
- **software_detected**: []
- **tables_count**: 1
- **PMC-first strategy**: ✅ Working correctly

**BMC** - PMID:33388025
- **Status**: ✅ Success
- **tier_used**: `full_pmc_xml`
- **source_type**: `pmc_xml`
- **extraction_time**: 2.83 seconds
- **methods_text_length**: 1476 chars
- **software_detected**: ['r']
- **tables_count**: 0
- **PMC-first strategy**: ✅ Working correctly

**PLOS (via DOI)** - 10.1371/journal.pone.0245093
- **Status**: ✅ Success
- **tier_used**: `full_pmc_xml`
- **source_type**: `pmc_xml`
- **extraction_time**: 4.40 seconds
- **methods_text_length**: 3857 chars
- **software_detected**: []
- **tables_count**: 0
- **PMC-first strategy**: ✅ Working correctly

**BMC (via DOI)** - 10.1186/s12864-020-07352-1
- **Status**: ❌ Failed
- **tier_used**: ``
- **source_type**: ``
- **extraction_time**: 0.82 seconds
- **methods_text_length**: 0 chars
- **software_detected**: []
- **tables_count**: 0
- **PMC-first strategy**: ❌ Not working
- **Error**: Paper 10.1186/s12864-020-07352-1 is paywalled. ## Alternative Access Options

2. **Preprint Servers**: Check for preprints on bioRxiv or medRxiv:
   - bioRxiv: https://www.biorxiv.org/search/10.1186/s12864-020-07352-1
   - medRxiv: https://www.medrxiv.org/search/10.1186/s12864-020-07352-1

3. **Institutional Access**: If you're affiliated with a university, try:
   - Accessing through your institution's library proxy
   - Using VPN to connect to institutional network
   - Requesting through interlibrary loan

5. **Unpaywall Service**: Check for legal open access versions:
   - https://unpaywall.org/10.1186/s12864-020-07352-1


💡 **Tip**: Many publishers allow authors to share accepted manuscripts. Contacting the corresponding author is often successful!

### Integration Summary:

- **Total tests**: 7
- **Successful extractions**: 6/7 (85.7%)
- **PMC XML used**: 6/7 (85.7%)
- **Fallback used**: 0/7 (0.0%)
- **PMC-first strategy working**: 6/7 (85.7%)

**Performance (PMC XML)**:
- Average extraction time: 2.91 seconds
- Min: 2.24s, Max: 4.40s

### Key Findings:

⚠️ **PMC-first strategy partially working**: 6/7 tests working correctly

