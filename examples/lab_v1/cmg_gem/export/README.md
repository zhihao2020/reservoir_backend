# GEM export drop

Put GEM Results here after the alignment deck runs.

```text
observations.csv    # inversion
controls.csv        # inversion
hidden/meta.json    # scoring only
hidden/*.npy
```

Do not copy hidden arrays into the observation CSV. The invert script refuses a `--hidden` argument on the invert-only path.
