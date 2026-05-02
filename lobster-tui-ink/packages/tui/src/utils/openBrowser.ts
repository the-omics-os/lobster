/**
 * Open a URL or file in the default browser.
 * Works on macOS (open), Linux (xdg-open), and Windows (start).
 */

import { spawn } from "child_process";

export function openInBrowser(target: string): void {
  const platform = process.platform;
  let cmd: string;
  let args: string[];

  switch (platform) {
    case "darwin":
      cmd = "open";
      args = [target];
      break;
    case "win32":
      cmd = "cmd";
      args = ["/c", "start", "", target];
      break;
    default:
      cmd = "xdg-open";
      args = [target];
      break;
  }

  const child = spawn(cmd, args, {
    detached: true,
    stdio: "ignore",
  });
  child.unref();
}
