# protonate_utils

A single CLI + Python library for adding hydrogens to ligands (RDKit +
Dimorphite-DL) and proteins (Biotite + Hydride) at a target pH. The whole
package is the single module `protonate_utils.py`, with tests in
`test_protonate_utils.py`.

## Releasing to PyPI

**Always publish through the GitHub Actions release flow — never run a manual
`twine upload` from a local checkout.** Publishing uses PyPI Trusted Publishing
(OIDC) configured in `.github/workflows/publish.yml`, which builds and uploads
automatically when a GitHub Release is *published*. No API token is stored
locally, so a manual upload would fail anyway.

To cut a release:

1. Bump `version` in `pyproject.toml` (e.g. `0.1.3` → `0.1.4`). Check the
   current value first and ensure the new one is a correct, strictly
   increasing increment over the last released version — compare against
   the latest tag (`git tag --list`) and the live PyPI version (see step 5)
   so you neither skip nor reuse a number. Confirm the exact version with
   the user before tagging — a PyPI version cannot be reused once published,
   even after deletion.
2. Commit the bump and push to `main`:
   ```bash
   git commit -am "Bump version to X.Y.Z" && git push origin main
   ```
3. Create the release, which also creates the `vX.Y.Z` tag (tags follow the
   `v`-prefixed convention) and triggers the publish workflow:
   ```bash
   gh release create vX.Y.Z --target main --title "vX.Y.Z" --notes "..."
   ```
4. Watch the workflow to completion and confirm success:
   ```bash
   gh run watch <run-id> --exit-status   # find id via: gh run list --workflow=publish.yml
   ```
5. Verify the new version is live:
   ```bash
   curl -sS https://pypi.org/pypi/protonate-utils/json \
     | python -c "import sys,json; print(json.load(sys.stdin)['info']['version'])"
   ```

## Testing

```bash
python -m pytest test_protonate_utils.py -q     # or: python test_protonate_utils.py
```

Protein-mode tests need `biotite` + `hydride`; they skip cleanly if absent.
