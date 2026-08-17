# Semantic checklists

The benchmark's other metrics are mechanical: they can tell you a model
answered, in the right script, keeping the numbers. They cannot tell you the
translation is *correct*. `aya-expanse-8b` passes all of them and still renders
Spanish *gracias* as Bulgarian "please".

A checklist closes that gap for one language pair. It is a list of things the
source actually says, each with a pattern the translation must match — and,
where a specific wrong answer is known, a pattern it must not. Every check is
scoped to one segment, so a score is auditable line by line rather than a
number you have to trust.

**This is the highest-value contribution to the benchmark.** Each checklist
roughly doubles what it can decide for that language. Writing one needs no
Python — only that you read the target language.

## File naming

```
<fixture-id>.<target-lang>.json
```

e.g. `es_mothers_day.bg.json` scores translations of the `es_mothers_day`
fixture into Bulgarian.

## Format

```json
{
  "fixture": "es_mothers_day",
  "target_lang": "bg",
  "author_note": "Checked by hand against the Spanish source.",
  "checks": [
    {
      "segment": 0,
      "label": "gracias -> благодаря",
      "expect": "благодар",
      "reject": "\\bМоля\\b",
      "why": "Моля is 'please', never 'gracias'"
    }
  ],
  "penalties": [
    {
      "label": "clitic order",
      "pattern": "(?:^|[.!?]\\s+)(?:Те|Ме|Ти|Се)\\s+\\w",
      "why": "Bulgarian clitics cannot open a clause: 'Обичам те', never 'Те обичам'"
    }
  ]
}
```

- `segment` is a 0-based index into the fixture's `segments`.
- `expect` is a regex that must match.
- `reject` is optional; if it matches, the check fails even when `expect` did.
- `penalties` are counted across the whole translation rather than passed or
  failed per segment. Use them for grammar errors a model makes repeatedly.
- `why` is for the human reading a failure. Always write one.

**Every pattern is matched case-insensitively**, `penalties` included. Write
them lowercase. This matters more than it sounds: the failure a penalty is
usually looking for — a model answering in the wrong language — produces
sentence-initial words, so a case-sensitive `\bздравей\b` would miss the
`Здравей` it exists to catch.

## Writing a good check

**Accept every correct answer, not just the one you thought of.** The first
version of the Bulgarian checklist rejected "Майчин ден" — which is the *more*
idiomatic rendering of Mother's Day than the one it was looking for — plus two
other valid variants, and mis-ranked the models as a result. When in doubt,
make `expect` looser and lean on `reject` for the specific wrong answer you
have actually seen.

**Check meaning, not style.** A stiff translation is not a failed one. Score
what changes what the viewer understands: a swapped subject, a wrong number, a
word that does not exist in the language.

**Prefer a stem to a full form.** `благодар` matches благодаря, благодарим and
благодаря ти. Matching the exact inflection you happened to see makes the check
brittle.
