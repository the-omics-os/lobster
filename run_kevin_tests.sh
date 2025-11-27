#!/bin/bash
# Automated test script for Kevin's bugs
# Generated: 2025-11-25 20:12:04

# Ensure Lobster is activated
# source .venv/bin/activate

# Test 1: Bug #2 Test 1 - GSE182227
echo '============================================================'
echo 'Test 1: Bug #2 Test 1 - GSE182227'
echo '============================================================'
lobster query "ADMIN SUPERUSER: Download and validate single-cell dataset GSE182227. Load it and report the shape (n_obs × n_vars). Check if variables are present."
echo '\nTest 1 complete. Waiting 5 seconds...'
sleep 5

# Test 2: Bug #2 Test 2 - GSE190729
echo '============================================================'
echo 'Test 2: Bug #2 Test 2 - GSE190729'
echo '============================================================'
lobster query "ADMIN SUPERUSER: Download and validate single-cell dataset GSE190729. This is a known reliable dataset. Load it and confirm successful loading."
echo '\nTest 2 complete. Waiting 5 seconds...'
sleep 5

# Test 3: Bug #3 Test 1 - GSE130036 Kallisto
echo '============================================================'
echo 'Test 3: Bug #3 Test 1 - GSE130036 Kallisto'
echo '============================================================'
lobster query "ADMIN SUPERUSER: Download bulk RNA-seq dataset GSE130036. This contains Kallisto quantification files. Load the data and check the orientation - report if samples are rows and genes are columns."
echo '\nTest 3 complete. Waiting 5 seconds...'
sleep 5

# Test 4: Bug #3 Test 2 - GSE130970 Metadata
echo '============================================================'
echo 'Test 4: Bug #3 Test 2 - GSE130970 Metadata'
echo '============================================================'
lobster query "ADMIN SUPERUSER: Download bulk RNA-seq dataset GSE130970 with 78 liver samples. Load it and verify the metadata includes clinical labels (Normal/NAFLD/NASH). Check if the orientation is correct (samples × genes)."
echo '\nTest 4 complete. Waiting 5 seconds...'
sleep 5

# Test 5: Bug #7 Test - FTP Download
echo '============================================================'
echo 'Test 5: Bug #7 Test - FTP Download'
echo '============================================================'
lobster query "ADMIN SUPERUSER: Test FTP download reliability by downloading GSE130036 using SUPPLEMENTARY_FIRST strategy. Report any timeout or connection errors during download."
echo '\nTest 5 complete. Waiting 5 seconds...'
sleep 5

