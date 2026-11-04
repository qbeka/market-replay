# Data rights

This repository contains no licensed market data. Every number in it comes from the seeded synthetic feed.

## What is in Git

| Content | In Git | Why |
|---|---|---|
| `data/fixtures/` | Yes | Hand-written, a few lines each |
| `data/synthetic/` | No | About 534 MB. `mreplay generate` rebuilds it byte for byte, and `index.json` holds each day's SHA-256 |
| `data/licensed/` | No | Git-ignored so licensed files cannot be committed by accident |

## Why no real data was used

The plan was to reconstruct a real window from LOBSTER's free sample files. That is no longer possible without your action. Checked on 4 November 2026:

- LOBSTER's free sample files are gone from the routes that served them. Samples now need proof of a book purchase, or proof of conference attendance, then staff approval.
- Access means accepting LOBSTER's [Trial and Sample Data Terms](https://lobsterdata.com/terms/trial-and-sample-data), version 2026-08-14. Accepting them is a binding agreement in your name, so this project did not do it.
- Clause 5.1(b) forbids publicly sharing "results, tables, figures, statistics, screenshots, or findings" derived from the data, without a full licence.

Even with a sample, anything derived from it could not appear on your website or resume. Read the terms yourself before relying on this summary. It is not legal advice.

## If you want real data later

| Source | What it gives you | What it needs from you |
|---|---|---|
| LOBSTER licence | Message and orderbook files, handled by `adapters/lobster.py` | An academic or paid licence |
| Databento MBO | Order-by-order feed with venue sequence numbers | An account, a paid order, and a new adapter |

Put the files in `data/licensed/`. Check the licence before you commit or publish anything derived from them.

## Format sources

LOBSTER's own format page rendered blank in this environment. The adapter's format comes from two secondary sources that quote it:

- [Converting LOBSTER demo R code into Python](https://www.r-bloggers.com/2019/12/converting-lobster-demo-r-code-into-python)
- [High-frequency econometrics, University of Copenhagen](https://aef-course.ku.dk/exercises/high-frequency-econometrics.html)

They disagree once. The first gives the buy direction as -2 and the second as 1. The adapter uses 1 and raises an error on any other value, so a real file would fail loudly if this is wrong.

The adapter has never read a real LOBSTER file. Treat it as tested against the format, not proven on the data.
