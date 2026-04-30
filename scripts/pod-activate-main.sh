#!/bin/bash
# Source this to revert to the main (shared) deeptools + torch-spyre.
# Usage: source pod-activate-main.sh

export DEEPTOOLS_INSTALL_DIR="$DTI_PROJECT_ROOT/sentient/deeptools"
export DEEPTOOLS_PATH="$DEEPTOOLS_INSTALL_DIR/share"
export PATH="$DEEPTOOLS_INSTALL_DIR/bin:$PATH"

# Reinstall main torch-spyre
cd "$DTI_PROJECT_ROOT/torch-spyre"
uv pip install -e . --reinstall-package torch_spyre 2>/dev/null

echo "Main environment active:"
echo "  deeptools: $(which dxp_standalone)"
echo "  torch-spyre: $(python3 -c 'import torch_spyre; print(torch_spyre.__file__)')"
