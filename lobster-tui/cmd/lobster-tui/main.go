// Package main is the entry point for lobster-tui, the Charm-based TUI
// frontend for Lobster AI.
package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/the-omics-os/lobster-tui/internal/chat"
	initwizard "github.com/the-omics-os/lobster-tui/internal/initwizard"
)

// Version is the current version of lobster-tui.
const Version = "0.1.0"

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "init":
		runInit(os.Args[2:])
	case "chat":
		runChat(os.Args[2:])
	case "version", "--version", "-v":
		fmt.Printf("lobster-tui %s\n", Version)
	case "help", "--help", "-h":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Printf(`lobster-tui %s — Charm TUI frontend for Lobster AI

Usage:
  lobster-tui <command> [flags]

Commands:
  init     Configure Lobster AI (API keys, theme, preferences)
  chat     Start interactive chat session
  version  Print version information
  help     Show this help message

Flags for init:
  --theme <name>        Theme to use (lobster-dark, lobster-light). Default: lobster-dark
  --result-file <path>  Write wizard result JSON to a file instead of stdout

Flags for chat:
  --proto-fd-in <fd>   File descriptor for protocol input (required)
  --proto-fd-out <fd>  File descriptor for protocol output (required)
  --theme <name>       Theme to use (lobster-dark, lobster-light). Default: lobster-dark
  --inline             Run in inline mode (no alternate screen, cleaner rendering)
`, Version)
}

// runInit parses the flags for the init subcommand and launches the wizard.
func runInit(args []string) {
	themeName := "lobster-dark"
	resultFile := ""

	// Parse --theme flag from args.
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--theme":
			if i+1 < len(args) {
				themeName = args[i+1]
				i++
			} else {
				fmt.Fprintf(os.Stderr, "error: --theme requires a value\n")
				os.Exit(1)
			}
		case "--result-file":
			if i+1 < len(args) {
				resultFile = args[i+1]
				i++
			} else {
				fmt.Fprintf(os.Stderr, "error: --result-file requires a value\n")
				os.Exit(1)
			}
		default:
			fmt.Fprintf(os.Stderr, "warning: unknown flag %q (ignored)\n", args[i])
		}
	}

	if err := initwizard.Run(themeName, resultFile); err != nil {
		// "cancelled" is a clean exit — exit code 1, no error message.
		if err.Error() == "cancelled" {
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "Init failed: %v\n", err)
		os.Exit(1)
	}
}

// runChat parses the flags for the chat subcommand and launches the session.
func runChat(args []string) {
	themeName := "lobster-dark"
	fdIn := -1
	fdOut := -1
	inline := false

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--proto-fd-in":
			if i+1 >= len(args) {
				fmt.Fprintf(os.Stderr, "error: --proto-fd-in requires a value\n")
				os.Exit(1)
			}
			v, err := strconv.Atoi(args[i+1])
			if err != nil {
				fmt.Fprintf(os.Stderr, "error: --proto-fd-in must be an integer: %v\n", err)
				os.Exit(1)
			}
			fdIn = v
			i++

		case "--proto-fd-out":
			if i+1 >= len(args) {
				fmt.Fprintf(os.Stderr, "error: --proto-fd-out requires a value\n")
				os.Exit(1)
			}
			v, err := strconv.Atoi(args[i+1])
			if err != nil {
				fmt.Fprintf(os.Stderr, "error: --proto-fd-out must be an integer: %v\n", err)
				os.Exit(1)
			}
			fdOut = v
			i++

		case "--theme":
			if i+1 >= len(args) {
				fmt.Fprintf(os.Stderr, "error: --theme requires a value\n")
				os.Exit(1)
			}
			themeName = args[i+1]
			i++

		case "--inline":
			inline = true

		default:
			fmt.Fprintf(os.Stderr, "warning: unknown flag %q (ignored)\n", args[i])
		}
	}

	if fdIn < 0 || fdOut < 0 {
		fmt.Fprintf(os.Stderr, "error: --proto-fd-in and --proto-fd-out are required\n\n")
		printUsage()
		os.Exit(1)
	}

	if err := chat.Run(fdIn, fdOut, themeName, Version, inline); err != nil {
		fmt.Fprintf(os.Stderr, "Chat failed: %v\n", err)
		os.Exit(1)
	}
}
