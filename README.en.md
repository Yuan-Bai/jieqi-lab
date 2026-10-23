# Jieqi Lab · Jieqi AI and self-play experiments

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/Yuan-Bai/jieqi-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Yuan-Bai/jieqi-lab/actions/workflows/ci.yml)

> An MIT-licensed toolkit for Jieqi (Chinese chess with hidden pieces): a Python rules engine,
> RIS-ISMCTS search, AlphaZero-style self-play training, a local browser UI, and a Tkinter GUI.
> Models, training data, and raw experiment artifacts are not distributed.

中文版：[README.md](README.md)

## Screenshots

Browser UI (local standard-library server; board, captures, analysis, and record panels):

![Browser UI](docs/images/web.png)

Desktop GUI (Tkinter; human/AI play, replay, and variation study):

<img src="docs/images/gui.png" alt="Tkinter desktop GUI" width="420">

## What is Jieqi?

Jieqi is a Chinese chess variant: every piece except the kings starts face down. A dark piece
moves according to the piece type of its starting square and is revealed after it moves. Each
player also has a private pool of unrevealed pieces. The game therefore combines perfect public
movement with symmetric hidden information.

The rules here are an independent implementation based on publicly published Jieqi rules. It is
an approximation, not an official referee, and is not affiliated with or endorsed by Tencent or
Tianxiang Qiyuan. `RULESET_VERSION = 5`.

## Features

- `jieqi.py`: rules engine, legal moves, long-check/chase rules, repetition, FEN, outcomes.
- `ismcts.py`: RIS-ISMCTS with belief tracking and root-conditioned determinization.
- `azero.py`: AlphaZero-style self-play, complete training checkpoints, run provenance, evaluation.
- `search_engine.py`: reusable engine adapter used by the web UI and analysis tools.
- `web_server.py` + `web/`: local browser game, analysis, replay, import/export; standard library
  only for RIS play.
- `gui.py`: Tkinter human/AI play, replay tree, variation study, position editor, sound.
- `pika_engine.py`: optional UCI bridge for an external Pikafish `jieqi_old` build.
- `tools/`: evaluation, benchmarks, diagnostics, provenance, and isolated experiments.
- `docs/`: ADRs, technical reports, research notes, and release history.

## Quick start

```bash
python web_server.py
```

Open <http://127.0.0.1:8765>. RIS-ISMCTS play needs only Python 3.11+ and the standard library.
The page supports human vs. AI, local two-player, AI vs. AI, hints, undo, branch replay, record
import/export, and synthesized sound. The server binds to `127.0.0.1` by default and validates
Host/Origin on every request; it is not intended for public deployment.

Tkinter GUI:

```bash
python gui.py
```

Neural-network features require NumPy and PyTorch 2.10+:

```bash
python -m pip install -r requirements.txt
```

The repository does not ship pretrained weights. Only load models and pickle data from sources you
trust; see [SECURITY.md](SECURITY.md).

A synthetic record with branches is available at
[examples/synthetic-branches.json](examples/synthetic-branches.json); import it in the web UI or GUI
without any model.

## Tests

```bash
python -m pytest -q -rs
node --check web/app.js
node --test web/preferences.test.cjs web/history.test.cjs web/view-state.test.cjs
```

Optional native extensions (Cython) are skipped when not built; that is not counted as a pass.
CI runs the standard-library/Web suite, the full CPU suite on Linux and Windows, and the native
backends on Linux. See [.github/workflows/ci.yml](.github/workflows/ci.yml) and
[CONTRIBUTING.md](CONTRIBUTING.md).

## Documentation

Start with [docs/README.md](docs/README.md):

- [status.md](docs/status.md): current engineering state and compatibility.
- [research-status.md](docs/research-status.md): research conclusions and open limits.
- [performance.md](docs/performance.md): acceleration options, builds, and fallbacks.
- [evaluation.md](docs/evaluation.md): frozen opponent pools and paired evaluation.
- [web.md](docs/web.md): browser UI and HTTP API.
- [gui.md](docs/gui.md): replay tree, variations, and position editor.
- [CONTEXT.md](CONTEXT.md): domain glossary and version constants.
- [docs/adr](docs/adr): architecture decision records.
- [docs/reviews](docs/reviews): dated experiment reports and audits.

Historical reports describe experiments in a private workspace. Their inputs, models, and Git
commits are not distributed; numbers are not current strength or performance guarantees.

## Security and privacy

- `torch.load` is always called with `weights_only=True`; PyTorch 2.10+ is required because of
  [GHSA-63cw-57p8-fm3p](https://github.com/advisories/GHSA-63cw-57p8-fm3p).
- Pickle data (`.pkl.gz`) is only for trusted sources; see [SECURITY.md](SECURITY.md).
- The web server is local-only: no authentication, TLS, remote rooms, or public deployment support.
- Browser preferences contain settings only; game records and private piece identities stay out of
  `localStorage`.
- Full GUI records contain private capture information. Review before sharing.

Report vulnerabilities through GitHub's private vulnerability reporting when enabled; otherwise
open a contact request without exploit details.

## License

Released under the [MIT License](LICENSE). Third-party dependencies and assets are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The project name and rules implementation are
independent; no upstream endorsement is implied.
