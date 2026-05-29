### What happened?

{{DESCRIPTION}}

### What did you expect to happen?

{{EXPECTED}}

### How can we reproduce it?

```python
import torch
import torch_spyre  # noqa: F401

device = torch.device("spyre")
# Minimal reproducer
{{REPRODUCER}}
```

### Any environmental details we need to know?

- PyTorch version: {{PYTORCH_VERSION}}
- torch-spyre version/commit: {{TORCH_SPYRE_VERSION}}
- Python version: {{PYTHON_VERSION}}
- OS: {{OS}}
- Spyre firmware version: {{FIRMWARE_VERSION}}
- Number of cores (`SENCORES`): {{SENCORES}}

### Anything else we need to know?

{{ADDITIONAL}}

### Relevant log output

```shell
{{LOGS}}
```
