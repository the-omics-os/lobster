package initwizard

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
)

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

// oauthSucceededMsg is sent when browser OAuth completes successfully.
// The Go TUI only receives tokens from the browser callback — email and
// tier are NOT sent by the frontend. Python postprocessing validates the
// token against the gateway and enriches credentials.
type oauthSucceededMsg struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	IDToken      string `json:"id_token"`
	ExpiresIn    int    `json:"expires_in"` // seconds, typically 3600
	Endpoint     string `json:"-"`          // not from browser; set locally
}

// oauthFailedMsg is sent when browser OAuth fails.
type oauthFailedMsg struct {
	Reason string // "timeout", "server_error", "cancelled", "insecure_endpoint"
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// oauthEndpoint returns the Omics-OS Cloud endpoint, respecting the
// OMICS_OS_ENDPOINT env var. Defaults to https://app.omics-os.com.
func oauthEndpoint() string {
	if ep := os.Getenv("OMICS_OS_ENDPOINT"); ep != "" {
		return ep
	}
	return "https://app.omics-os.com"
}

// isSecureEndpoint checks that the endpoint is HTTPS or localhost (for dev).
// Mirrors Python's _is_secure_endpoint in cloud_commands.py.
func isSecureEndpoint(endpoint string) bool {
	u, err := url.Parse(endpoint)
	if err != nil {
		return false
	}
	if u.Scheme == "https" {
		return true
	}
	if u.Scheme == "http" {
		host := u.Hostname()
		if host == "localhost" || host == "127.0.0.1" || host == "::1" {
			return true
		}
	}
	return false
}

// corsOrigin extracts the browser origin (scheme://host) from an endpoint URL.
// Browsers send Origin headers without paths or trailing slashes, so CORS
// Access-Control-Allow-Origin must match exactly.
func corsOrigin(endpoint string) string {
	u, err := url.Parse(endpoint)
	if err != nil {
		return endpoint
	}
	return fmt.Sprintf("%s://%s", u.Scheme, u.Host)
}

// randomState generates a cryptographically random URL-safe state string.
// Panics if the OS CSPRNG is unavailable — never fall back to predictable values.
func randomState() string {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		panic(fmt.Sprintf("crypto/rand failed: %v", err))
	}
	return base64.RawURLEncoding.EncodeToString(b)
}

// openBrowser opens a URL in the user's default browser.
func openBrowser(rawURL string) error {
	switch runtime.GOOS {
	case "darwin":
		return exec.Command("open", rawURL).Start()
	case "linux":
		return exec.Command("xdg-open", rawURL).Start()
	case "windows":
		// Use rundll32 instead of cmd /c start to avoid shell interpretation
		// of & characters in the URL (which cmd.exe treats as command separators).
		return exec.Command("rundll32", "url.dll,FileProtocolHandler", rawURL).Start()
	default:
		return fmt.Errorf("unsupported platform: %s", runtime.GOOS)
	}
}

// setCORSHeaders sets CORS and Private Network Access headers on the response.
// Required because the browser at https://app.omics-os.com makes cross-origin
// requests to http://localhost (different origin + private network).
// Origin is extracted from the endpoint URL (scheme://host only, no path).
func setCORSHeaders(w http.ResponseWriter, origin string) {
	w.Header().Set("Access-Control-Allow-Origin", origin)
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	// Chrome/Firefox Private Network Access (PNA) — required for
	// public origin → localhost cross-origin requests.
	w.Header().Set("Access-Control-Allow-Private-Network", "true")
}

// ---------------------------------------------------------------------------
// OAuth flow setup + tea.Cmd
// ---------------------------------------------------------------------------

// oauthFlowParams holds pre-bound resources for the OAuth flow.
// Created by prepareOAuthFlow in the main goroutine so the auth URL is
// available for display BEFORE the background Cmd starts.
type oauthFlowParams struct {
	Listener net.Listener
	Port     int
	State    string
	Endpoint string
	AuthURL  string
	Origin   string // CORS origin (scheme://host)
}

// prepareOAuthFlow binds an ephemeral port, generates CSRF state, and
// constructs the auth URL. Called from the main goroutine (wizard model)
// so that oauthAuthURL can be displayed to the user immediately.
// Returns nil and an error reason string if preparation fails.
func prepareOAuthFlow() (*oauthFlowParams, string) {
	endpoint := oauthEndpoint()

	if !isSecureEndpoint(endpoint) {
		return nil, "insecure_endpoint"
	}

	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		return nil, "server_error"
	}

	port := listener.Addr().(*net.TCPAddr).Port
	state := randomState()
	authURL := fmt.Sprintf("%s/auth/cli?port=%d&state=%s", endpoint, port, state)

	return &oauthFlowParams{
		Listener: listener,
		Port:     port,
		State:    state,
		Endpoint: endpoint,
		AuthURL:  authURL,
		Origin:   corsOrigin(endpoint),
	}, ""
}

// startOAuthFlow returns a BubbleTea Cmd that runs the browser OAuth flow
// using pre-bound resources from prepareOAuthFlow.
//
// Credentials are NOT saved inside this Cmd — the returned oauthSucceededMsg
// carries all tokens, and the wizard model saves them after accepting the message.
func startOAuthFlow(params *oauthFlowParams, cancel <-chan struct{}) tea.Cmd {
	return func() tea.Msg {
		resultCh := make(chan oauthSucceededMsg, 1)
		errCh := make(chan error, 1)

		mux := http.NewServeMux()
		mux.HandleFunc("POST /callback", func(w http.ResponseWriter, r *http.Request) {
			// CORS headers on the actual response (not just preflight).
			setCORSHeaders(w, params.Origin)

			ct := r.Header.Get("Content-Type")
			if !strings.Contains(ct, "application/json") {
				w.WriteHeader(415)
				return
			}

			// Limit body size to prevent abuse from local processes.
			r.Body = http.MaxBytesReader(w, r.Body, 64*1024)

			// Match the frontend payload from CliAuthCallback.tsx:
			// { id_token, access_token, refresh_token, expires_in, state }
			var payload struct {
				State        string `json:"state"`
				AccessToken  string `json:"access_token"`
				RefreshToken string `json:"refresh_token"`
				IDToken      string `json:"id_token"`
				ExpiresIn    int    `json:"expires_in"`
			}
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(400)
				_, _ = w.Write([]byte(`{"error":"invalid_json"}`))
				return
			}

			if payload.State != params.State {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(403)
				_, _ = w.Write([]byte(`{"error":"state_mismatch"}`))
				return
			}

			// Require non-empty access token.
			if strings.TrimSpace(payload.AccessToken) == "" {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(400)
				_, _ = w.Write([]byte(`{"error":"missing_access_token"}`))
				return
			}

			// Return JSON so the frontend fetch() can parse it cleanly.
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(200)
			_, _ = w.Write([]byte(`{"status":"ok"}`))

			// Non-blocking send — ignore duplicate callbacks.
			select {
			case resultCh <- oauthSucceededMsg{
				AccessToken:  payload.AccessToken,
				RefreshToken: payload.RefreshToken,
				IDToken:      payload.IDToken,
				ExpiresIn:    payload.ExpiresIn,
				Endpoint:     params.Endpoint,
			}:
			default:
			}
		})
		mux.HandleFunc("OPTIONS /callback", func(w http.ResponseWriter, r *http.Request) {
			setCORSHeaders(w, params.Origin)
			w.Header().Set("Access-Control-Max-Age", "300")
			w.WriteHeader(204)
		})

		server := &http.Server{
			Handler:           mux,
			ReadHeaderTimeout: 10 * time.Second,
			ReadTimeout:       15 * time.Second,
		}

		// Serve in background goroutine.
		go func() {
			if err := server.Serve(params.Listener); err != nil && err != http.ErrServerClosed {
				select {
				case errCh <- err:
				default:
				}
			}
		}()

		_ = openBrowser(params.AuthURL)

		// Wait for result, cancel, timeout, or server error.
		defer server.Close()

		select {
		case result := <-resultCh:
			return result
		case <-cancel:
			return oauthFailedMsg{Reason: "cancelled"}
		case <-errCh:
			return oauthFailedMsg{Reason: "server_error"}
		case <-time.After(120 * time.Second):
			return oauthFailedMsg{Reason: "timeout"}
		}
	}
}

// ---------------------------------------------------------------------------
// Credential persistence
// ---------------------------------------------------------------------------

// saveOAuthCredentials writes OAuth credentials to
// ~/.config/omics-os/credentials.json with mode 0600.
// This mirrors the Python credentials.py format.
//
// Note: email, tier, and user_id are NOT populated here — they come from
// gateway validation which happens in Python postprocessing.
func saveOAuthCredentials(msg oauthSucceededMsg) error {
	configDir := filepath.Join(homeDir(), ".config", "omics-os")
	if err := os.MkdirAll(configDir, 0700); err != nil {
		return fmt.Errorf("create config dir: %w", err)
	}
	if err := os.Chmod(configDir, 0700); err != nil {
		return fmt.Errorf("chmod config dir: %w", err)
	}

	expiresIn := msg.ExpiresIn
	if expiresIn <= 0 {
		expiresIn = 3600
	}

	creds := map[string]interface{}{
		"auth_mode":     "oauth",
		"access_token":  msg.AccessToken,
		"refresh_token": msg.RefreshToken,
		"id_token":      msg.IDToken,
		"endpoint":      msg.Endpoint,
		"token_expiry":  time.Now().UTC().Add(time.Duration(expiresIn) * time.Second).Format(time.RFC3339),
	}

	data, err := json.MarshalIndent(creds, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal credentials: %w", err)
	}
	data = append(data, '\n')

	credPath := filepath.Join(configDir, "credentials.json")

	f, err := os.OpenFile(credPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0600)
	if err != nil {
		return fmt.Errorf("open credentials file: %w", err)
	}
	defer f.Close()

	if err := f.Chmod(0600); err != nil {
		return fmt.Errorf("chmod credentials file: %w", err)
	}
	if _, err := f.Write(data); err != nil {
		return fmt.Errorf("write credentials: %w", err)
	}

	return nil
}

// homeDir returns the user's home directory.
func homeDir() string {
	if h, err := os.UserHomeDir(); err == nil {
		return h
	}
	return os.Getenv("HOME")
}
