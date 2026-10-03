---
name: incident-summary
description: Write a structured summary of a production incident or outage (impact, root cause, fix, follow-ups). Use when the user asks to summarize, explain or report on an incident or postmortem.
---

# Incident summary

## When to use
- The user asks what happened in an incident, outage or postmortem (e.g. "summarize INC-0917").

## Steps
1. Ask **engineering-agent** for the incident's timeline, root cause, resolution and customer impact.
2. Write the summary in exactly this format, using only facts the agent returned:

```
**Incident:** <id and one-line title>
**When / duration:** <date, time window, total minutes>
**Impact:** <who and what was affected>
**Root cause:** <one or two sentences>
**Fix:** <immediate fix, then permanent fix>
**Follow-ups:** <alerts, process changes, credits>
[<source file>]
```

3. If a field isn't in the documents, write "not documented". Don't guess.
