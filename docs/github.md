# GitHub workflow

Repository: https://github.com/HitarthSS/surgicalreal2sim2real

This is a standalone source repository. Research data, annotations, model weights and generated reconstructions remain external.

Before committing changes:

```bash
python3 -m unittest discover -s tests -v
python3 scripts/check_repository.py
python3 scripts/build_source_manifest.py
git diff --check
git status --short
```

The content checker catches common credential formats and unwanted model/data files. It is not a replacement for reviewing staged content. Keep real configurations and manifests in ignored `*.local.json` files.

Work on a branch, stage the intended changes, review them and open a pull request:

```bash
git switch -c codex/your-change
git add PATHS_TO_CHANGED_FILES
git diff --cached --stat
git diff --cached --check
git commit -m "Describe the change"
git push -u origin HEAD
```

The pipeline commands never run these Git operations themselves. No repository-wide license has been assigned; third-party notices and original terms remain applicable.
