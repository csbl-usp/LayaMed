# Contributing to LayaMed

Thanks for helping. LayaMed is a teaching example: changes that make the guide clearer, the code simpler or the
interface easier to use are the most welcome.

## Never share patient data

Do not paste real clinical notes, MIMIC text or anything derived from restricted data in issues, pull requests,
tests or example cases, even de-identified. Use the synthetic generator (`synthetic/generate_cases.py`) or write a
fictional case.

## Set up

```bash
git clone https://github.com/csbl-usp/LayaMed.git
cd LayaMed
python -m pip install -r requirements-dev.txt
python -m pytest
```

The tests need no GPU and no model. To run the interface you need a trained checkpoint: see
[docs/01-environment.md](docs/01-environment.md) and [docs/08-run-locally.md](docs/08-run-locally.md), or the Colab
notebook.

## Making a change

1. Open an issue first for anything larger than a small fix, so we can agree on the approach.
2. Keep training and inference in the same format: the notebook and the server import the same
   `layamed/inference.py`. A change to cleaning, sections or questions changes what the model reads, so say so in the
   pull request and re-run the notebook.
3. Match the style of the surrounding code (names, comment density, line length of about 120 characters).
   Code, comments and docs are in English.
4. Add or update tests in `tests/` for logic that runs without a GPU, and run `python -m pytest`.
5. Update the guide in `docs/` when behaviour changes.
6. Dependencies are pinned to tested minor versions in `requirements.txt`; change them only after re-running the
   notebook.

## Reporting a problem

Use the issue templates. For a security problem, follow [SECURITY.md](SECURITY.md) instead of opening a public issue.

By contributing, you agree that your contribution is licensed under the [MIT license](LICENSE) and that you follow
the [code of conduct](CODE_OF_CONDUCT.md).
