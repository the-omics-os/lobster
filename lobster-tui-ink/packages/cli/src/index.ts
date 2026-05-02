#!/usr/bin/env node
// @omicsos/cli — Omics-OS Cloud CLI (Commander-based one-shot commands)
// Phase 2 will populate this with session, files, export, agents, vault, editor commands
import { Command } from "commander";

const program = new Command();

program
  .name("lobster")
  .description("Omics-OS Cloud CLI")
  .version("1.1.410");

program.command("cloud")
  .description("Cloud operations")
  .action(() => { program.commands.find(c => c.name() === "cloud")?.outputHelp(); });

program.parse();
