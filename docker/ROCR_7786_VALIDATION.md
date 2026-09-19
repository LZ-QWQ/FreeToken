# ROCr #7786 validation

## Scope

- FreeToken branch: `validate/rocr-7786-backport`
- Base image digest: `sha256:65bfbe0c6affc15453abdb2870288c5d1cf8cfd207e9033ca473a982d36bfa77`
- ROCm systems base: `ca887ee80abfb82671fe1d6d8da708a713438e05` (`therock-7.14.1`)
- Backported fix: `096e6f1b729a7d6f68a1cd333eb565ed382e47ca` (ROCm/rocm-systems#7786)
- Validation image digest: `sha256:5f92571b4439760e5d26263976c24610b0671ccef9219a148188f30bdcd4b0c7`
- Host: W7900D, `gfx1100`, GPU 6

The validation image changes only ROCr. It replaces both ROCr copies used by
the base image:

- `/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1.21.0`
- `/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1`

The second path is the copy actually loaded by the ROCm PyTorch wheel. The
rebuilt library is linked against the same `librocm_sysdeps_*` sonames as the
ROCm 7.14.1 baseline.

## Results

All tests used Docker's default `RLIMIT_NOFILE.soft=1024`. No `--ulimit`
override was supplied.

The HIP VMM probe created 1,500 20 MiB handles (30,000 MiB total):

```text
chunk_mib=20 granularity=4096 free_before=50790924288 total=51522830336 open_fds_before=14 nofile_soft=1024 nofile_hard=524288
successful_handles=1500 successful_mib=30000 free_after_create=19333644288 open_fds_after_create=14
open_fds_after_release=14
```

The PyTorch probe set `PYTORCH_ALLOC_CONF=expandable_segments:True`, allocated
20 GiB, wrote the complete allocation with `zero_()`, synchronized the GPU,
and verified the allocator snapshot contained an expandable segment:

```text
torch=2.11.0+rocm7.14.1 hip=7.14.60850 nofile_soft=1024 nofile_hard=524288 free_before=50790924288 total=51522830336 open_fds_before=14
allocation_bytes=21474836480 free_after=29160898560 open_fds_after_allocation=15 expandable_segments=1
open_fds_after_release=15
```

Before the backport, the 20 MiB HIP probe failed on handle index 1011 because
one dma-buf FD remained open per handle and exhausted the 1024-FD soft limit.
The patched results show that FD usage is no longer proportional to the number
of VMM handles. Raising `nofile` is therefore unnecessary for this reproducer.
