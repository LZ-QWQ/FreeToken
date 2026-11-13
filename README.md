# FP8 bandwidth calibration validation

Validation evidence for `fix(moe): pad fp8 calibration scale banks like the loader`.

Code candidate: [f1e272b17ddc4e4280bcc22653ae74646bff4b96](https://github.com/LZ-QWQ/FreeToken/commit/f1e272b17ddc4e4280bcc22653ae74646bff4b96). Hardware comparison baseline: FreeToken main `27740fec662ec0f0ce9a74125162a905d35171e5`.

[Download the validation archive](benchbw-fp8-validation.zip) and verify its SHA256 against [the checksum](benchbw-fp8-validation.zip.sha256). It contains the measured benchmark driver, fixed 32K token inputs, original request records, environment versions, calibration profiles, numerical/copy validation summaries, and statistics for RTX4090, RTX5090, W7900D, R9700 and H100.

To recompute the 60 formal request samples without a GPU, unzip the archive and run from its root:

```sh
python scripts/analyze.py --data-root . --output recalculated
```

The archive README documents hardware reproduction, warmup exclusion, AB/BA sampling, Decode/Generate definitions, pooled results, and output-quality limitations. All formal samples are retained. Model weights are not included.

This branch contains validation artifacts only; the code PR changes only `python/freetoken/moe/benchbw.py`.
