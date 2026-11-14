# Managed Cloud Session Quick Start — npm Only

Use this route only for a managed Cloud session, not to set up local Python
Lobster AI using a Cloud account. These examples use npm CLI 0.4.2 and its
`lobster-cli` alias, not Python's binary:

```bash
npm install -g @omicsos/lobster@0.4.2  # only with permission
lobster-cli cloud login --device     # user authorizes the displayed URL/code
lobster-cli cloud status
lobster-cli cloud chat
lobster-cli cloud query --json "Describe the data available in this session"
```

Device login still needs a human browser approval; do not automate passwords or
copy credentials into prompts. Sign-in writes shared credentials under
`~/.config/omics-os/`; obtain permission for that location. For npm's headless
Cloud-session automation, provide `OMICSOS_API_KEY` through the environment.
Python's local gateway key uses `OMICS_OS_API_KEY` instead. npm login is not a
substitute for installing and configuring Python. A stored credential file is
not proof that authentication still works; inspect status from the selected client.
