## Summary

Describe the behavior change and the problem it solves.

## Scope and compatibility

- Affected area: rules / search / training / web / GUI / docs / tooling
- Version or protocol impact: `RULESET_VERSION`, `DATA_VERSION`, `SEARCH_VERSION`,
  `EVAL_PROTOCOL_VERSION`, replay format
- Hidden-information or checkpoint/provenance impact:

## Checks run

- [ ] `python -m pytest -q -rs` (or the most relevant subset, named below)
- [ ] `node --check web/app.js`
- [ ] `node --test web/preferences.test.cjs web/history.test.cjs web/view-state.test.cjs`
- [ ] Manual web/GUI check (describe)

Commands and results:

```text
paste concise output here
```

Skipped environments and why:

## Notes for reviewers

Link the issue/ADR, call out approximation limits, and state whether models, training data or
private replays were used. Do not attach private data, secrets or absolute local paths.
