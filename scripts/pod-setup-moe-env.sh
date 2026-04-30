#!/bin/bash
# One-time setup for MoE isolated testing environment on Spyre pod.
# Creates worktrees and builds for the moe-indirect-access branches of
# both torch-spyre and deeptools, isolated from the shared main installs.
#
# Usage: bash pod-setup-moe-env.sh
set -e
echo "=== Setting up MoE isolated environment ==="

# 1. Create deeptools worktree if not exists
if [ ! -d "$DTI_PROJECT_ROOT/deeptools-moe" ]; then
    echo "Creating deeptools worktree..."
    cd "$DTI_PROJECT_ROOT/deeptools"
    git fetch origin moe-indirect-access
    git worktree add "$DTI_PROJECT_ROOT/deeptools-moe" origin/moe-indirect-access
else
    echo "deeptools-moe worktree already exists, updating..."
    cd "$DTI_PROJECT_ROOT/deeptools-moe"
    git fetch origin moe-indirect-access
    git checkout --detach origin/moe-indirect-access
fi

# 2. Build deeptools from MoE branch into isolated install
echo "Building deeptools (MoE branch)..."
export MOE_BUILD="$DTI_PROJECT_ROOT/build/deeptools-moe"
export MOE_INSTALL="$DTI_PROJECT_ROOT/sentient/deeptools-moe"
mkdir -p "$MOE_BUILD" "$MOE_INSTALL"

cd "$MOE_BUILD"
cmake "$DTI_PROJECT_ROOT/deeptools-moe" \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_INSTALL_PREFIX="$MOE_INSTALL" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DDT_USE_DCC_DDC="on" \
    -DLLVM_PROJ_SRC="${LLVM_PROJ_SRC}" \
    -DLLVM_PROJ_BUILD="${LLVM_PROJ_BUILD}" \
    -DMANAGE_LLVM=0 \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1
make -j"${MAX_JOBS:-64}" install

# 3. Ensure torch-spyre-moe worktree exists
if [ ! -d "$DTI_PROJECT_ROOT/torch-spyre-moe" ]; then
    echo "Creating torch-spyre worktree..."
    cd "$DTI_PROJECT_ROOT/torch-spyre"
    git remote add fork https://github.com/raghukiran1224/torch-spyre.git 2>/dev/null || true
    git fetch fork worktree-moe-indirect-access
    git worktree add "$DTI_PROJECT_ROOT/torch-spyre-moe" fork/worktree-moe-indirect-access
else
    echo "torch-spyre-moe worktree already exists, updating..."
    cd "$DTI_PROJECT_ROOT/torch-spyre-moe"
    git fetch fork worktree-moe-indirect-access
    git checkout --detach fork/worktree-moe-indirect-access
fi

# 4. Install torch-spyre from MoE worktree
echo "Installing torch-spyre (MoE branch)..."
cd "$DTI_PROJECT_ROOT/torch-spyre-moe"
uv pip install -e . --reinstall-package torch_spyre

echo ""
echo "=== MoE environment setup complete ==="
echo "To activate: source $DTI_PROJECT_ROOT/torch-spyre-moe/scripts/pod-activate-moe.sh"
