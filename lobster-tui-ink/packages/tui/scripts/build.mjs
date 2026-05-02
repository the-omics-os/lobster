#!/usr/bin/env node
import { build } from "esbuild";

const result = await build({
  entryPoints: ["src/cli.tsx"],
  bundle: true,
  platform: "node",
  target: "node22",
  format: "esm",
  outfile: "dist/lobster.mjs",
  banner: {
    js: [
      "#!/usr/bin/env node",
      'import { createRequire } from "module";',
      "const require = createRequire(import.meta.url);",
    ].join("\n"),
  },
  define: {
    "process.env.NODE_ENV": '"production"',
    "process.env.DEV": '"false"',
  },
  jsx: "automatic",
  sourcemap: true,
  minify: true,
  metafile: true,
  logLevel: "info",
});

const outputs = result.metafile.outputs;
for (const [file, meta] of Object.entries(outputs)) {
  if (file.endsWith(".mjs")) {
    console.log(`\n${file}: ${(meta.bytes / 1024).toFixed(0)} KB`);
  }
}
