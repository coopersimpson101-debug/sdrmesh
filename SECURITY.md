# Security policy

## Supported versions

Only the latest commit on `main` is supported. sdrmesh is a research/teaching project and has not yet been run
against live radio traffic.

## Scope

sdrmesh parses untrusted radio input (BLE PDUs, LoRa headers and payloads, LR link frames) and untrusted files
(IQ captures, SigMF JSON). Please report privately, for example:

- crashes, hangs or unbounded memory/CPU use reachable from a crafted capture or over the air;
- parser bugs in `sdrmesh/ble/pdu.py`, `sdrmesh/lora/coding.py` or `sdrmesh/link.py` that accept malformed
  frames as valid or mis-attribute data;
- anything that makes the `rx` command run external programs other than the documented `hackrf_transfer`.

The meshfind cryptography (report encryption, rotating keys) is out of scope here; please report those issues
to the meshfind repository.

## Reporting a vulnerability

Please use **GitHub private vulnerability reporting**: open the repository's **Security** tab and click
**Report a vulnerability** (https://github.com/coopersimpson101-debug/sdrmesh/security/advisories/new).
Do not open a public issue for an unfixed vulnerability. Include a description, a reproducing capture or script,
and the commit hash you tested.

## What to expect

An acknowledgement within 7 days, an assessment within 30 days, and credit in the release notes if you want it.
