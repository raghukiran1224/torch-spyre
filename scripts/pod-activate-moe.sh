#!/bin/bash
# Source this to switch to the MoE branch of deeptools + torch-spyre.
# Usage: source pod-activate-moe.sh

export DEEPTOOLS_INSTALL_DIR="$DTI_PROJECT_ROOT/sentient/deeptools-moe"
export DEEPTOOLS_PATH="$DEEPTOOLS_INSTALL_DIR/share"
export PATH="$DEEPTOOLS_INSTALL_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$DEEPTOOLS_INSTALL_DIR/lib:${LD_LIBRARY_PATH:-}"

# Ensure torch-spyre-moe is the active install
cd "$DTI_PROJECT_ROOT/torch-spyre-moe"
uv pip install -e . --reinstall-package torch_spyre 2>/dev/null

echo "MoE environment active:"
echo "  deeptools: $(which dxp_standalone)"
echo "  torch-spyre: $(python3 -c 'import torch_spyre; print(torch_spyre.__file__)')"
