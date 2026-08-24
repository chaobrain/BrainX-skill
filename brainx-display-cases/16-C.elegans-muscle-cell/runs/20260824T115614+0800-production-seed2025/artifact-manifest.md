# Artifact manifest

| Artifact | SHA-256 | Mechanical check |
|---|---|---|
| `raw/report.json` | `f93aa45b1bb23ac84b45e9c81de9f0c3ebba08fe50d332c3d7eaef7627afb563` | Strict JSON parse passed |
| `raw/posterior_samples.csv` | `1631038ed015a8e6f846962af384870e447938f638b63cce6e6f49020436bb51` | 128 x 8, finite, weights sum to 1 within floating precision |
| `raw/recovery_results.csv` | `e344e4fc6eb7b867e26b46b68a0a7bd9b3bb907017ebcaa49c15b0112c91c656` | 3 x 14, finite |
| `raw/trace_predictions.csv` | `5fac48418f546ef4472c7deb5f6033ce4128f2e76c849067f110f42ee2bcc8bf` | 5,000 x 9, finite |
| `raw/held_out_validation.png` | `1a8acc9e6cd4e61040cbd5a850cf0ba4eb5df4ab0003080e6acd41903284d096` | 324,328 bytes, visually inspected |
| `run.log` | `fd3e257c0b2fd7ada76a1659d82f9d894359155cd82ba86f9fc01407d2bbf5bc` | Complete log with exit code 0 |

Post-run test suite: 6 tests passed in 8.741 s on the recorded CPU environment.
