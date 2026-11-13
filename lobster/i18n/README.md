# Adding a language

English (`en.json`) is the full list of UI strings. Another language is a JSON file in this directory. It may translate any subset of the English keys. It may not add keys.

1. Copy the keys you want from `en.json` into `xx.json`.
2. Use `{name}` placeholders exactly as English does.
3. `--lang xx` selects it. `LOBSTER_LANG` and `lang` in `.lobster_workspace/provider_config.json` do the same, in that order.
4. Unknown codes and missing keys show English.

The first overlay is `zh` (easy Simplified Chinese). These names also select it: `zh-CN`, `zh-Hans`, `cn`, `chinese`, `mandarin`.

Command names, flags, and `--json` keys stay English. Tool docstrings and agent prompts stay English. When the UI language is `zh`, the supervisor is asked to answer in easy Simplified Chinese and to leave tool names, paths, gene symbols, accessions, and code in English.

A new string that a local user reads is not done until it has a Chinese line. Use `t("dotted.key")` and add that key to `en.json` and `zh.json`, or use `h("English sentence")` and add the sentence to `phrases.zh.json`. Copy `en.json`, `zh.json`, and `phrases.zh.json` to `lobster-tui/internal/i18n/catalogs/`. The Go test fails if those copies differ.
