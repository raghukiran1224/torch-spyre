#!/bin/bash
# =============================================================================
# MoE Isolated Environment Setup for Spyre Pod
# =============================================================================
# Creates a fully isolated venv + build for the moe-indirect-access branches.
# Does NOT touch the shared .venv or sentient/deeptools installs.
#
# Usage:
#   bash pod-setup-moe-env.sh          # Full setup (first time)
#   bash pod-setup-moe-env.sh --quick  # Skip deeptools rebuild (just torch-spyre)
#
# After setup, activate with:
#   source $DTI_PROJECT_ROOT/moe-env/activate.sh
# =============================================================================
set -e

MOE_ROOT="$DTI_PROJECT_ROOT/moe-env"
MOE_VENV="$MOE_ROOT/venv"
MOE_DT_BUILD="$MOE_ROOT/build/deeptools"
MOE_DT_INSTALL="$MOE_ROOT/install/deeptools"
QUICK=0

for arg in "$@"; do
    case $arg in
        --quick) QUICK=1 ;;
    esac
done

echo "=== MoE Environment Setup ==="
echo "  Root:           $MOE_ROOT"
echo "  Venv:           $MOE_VENV"
echo "  DT Build:       $MOE_DT_BUILD"
echo "  DT Install:     $MOE_DT_INSTALL"
echo "  Quick mode:     $QUICK"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 1. Ensure worktrees exist and are up-to-date
# ─────────────────────────────────────────────────────────────────────────────
echo "[1/5] Updating source worktrees..."

# torch-spyre
if [ ! -d "$DTI_PROJECT_ROOT/torch-spyre-moe" ]; then
    cd "$DTI_PROJECT_ROOT/torch-spyre"
    git remote add fork https://github.com/raghukiran1224/torch-spyre.git 2>/dev/null || true
    git fetch fork worktree-moe-indirect-access
    git worktree add "$DTI_PROJECT_ROOT/torch-spyre-moe" fork/worktree-moe-indirect-access
else
    cd "$DTI_PROJECT_ROOT/torch-spyre-moe"
    git fetch fork worktree-moe-indirect-access 2>/dev/null || git fetch origin worktree-moe-indirect-access 2>/dev/null || true
    git checkout --detach FETCH_HEAD 2>/dev/null || true
fi
echo "  torch-spyre-moe: $(cd $DTI_PROJECT_ROOT/torch-spyre-moe && git log --oneline -1)"

# deeptools
if [ ! -d "$DTI_PROJECT_ROOT/deeptools-moe" ]; then
    cd "$DTI_PROJECT_ROOT/deeptools"
    git fetch origin moe-indirect-access
    git worktree add "$DTI_PROJECT_ROOT/deeptools-moe" origin/moe-indirect-access
else
    cd "$DTI_PROJECT_ROOT/deeptools-moe"
    git fetch origin moe-indirect-access 2>/dev/null || true
    git checkout --detach origin/moe-indirect-access 2>/dev/null || true
fi
echo "  deeptools-moe: $(cd $DTI_PROJECT_ROOT/deeptools-moe && git log --oneline -1)"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Create isolated venv
# ─────────────────────────────────────────────────────────────────────────────
echo "[2/5] Setting up isolated venv..."
mkdir -p "$MOE_ROOT"

if [ ! -d "$MOE_VENV" ]; then
    python3 -m venv "$MOE_VENV" --system-site-packages
    echo "  Created new venv at $MOE_VENV"
else
    echo "  Venv already exists at $MOE_VENV"
fi

# Activate venv for the rest of this script
source "$MOE_VENV/bin/activate"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Build deeptools from MoE branch
# ─────────────────────────────────────────────────────────────────────────────
if [ "$QUICK" -eq 0 ]; then
    echo "[3/5] Building deeptools (MoE branch)..."
    mkdir -p "$MOE_DT_BUILD" "$MOE_DT_INSTALL"

    cd "$MOE_DT_BUILD"
    cmake "$DTI_PROJECT_ROOT/deeptools-moe" \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_INSTALL_PREFIX="$MOE_DT_INSTALL" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DDT_USE_DCC_DDC="on" \
        -DLLVM_PROJ_SRC="${LLVM_PROJ_SRC}" \
        -DLLVM_PROJ_BUILD="${LLVM_PROJ_BUILD}" \
        -DMANAGE_LLVM=0 \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
        > /dev/null 2>&1
    make -j"${MAX_JOBS:-64}" install > /dev/null 2>&1
    echo "  Deeptools installed to $MOE_DT_INSTALL"
else
    echo "[3/5] Skipping deeptools build (--quick mode)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Install torch-spyre from MoE branch
# ─────────────────────────────────────────────────────────────────────────────
echo "[4/5] Installing torch-spyre (MoE branch)..."
cd "$DTI_PROJECT_ROOT/torch-spyre-moe"
pip install -e . --no-deps --no-build-isolation > /dev/null 2>&1
echo "  torch-spyre installed from $DTI_PROJECT_ROOT/torch-spyre-moe"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Create activation script
# ─────────────────────────────────────────────────────────────────────────────
echo "[5/5] Creating activation script..."
cat > "$MOE_ROOT/activate.sh" << 'ACTIVATE_EOF'
#!/bin/bash
# Source this to activate the MoE isolated environment.
# Usage: source $DTI_PROJECT_ROOT/moe-env/activate.sh

_MOE_ROOT="$DTI_PROJECT_ROOT/moe-env"

# Activate venv
source "$_MOE_ROOT/venv/bin/activate"

# Point to MoE deeptools install
export DEEPTOOLS_INSTALL_DIR="$_MOE_ROOT/install/deeptools"
export DEEPTOOLS_PATH="$DEEPTOOLS_INSTALL_DIR/share"
export PATH="$DEEPTOOLS_INSTALL_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$DEEPTOOLS_INSTALL_DIR/lib:${LD_LIBRARY_PATH:-}"

# Clean inductor cache (stale artifacts cause confusing failures)
rm -rf /tmp/torchinductor_$(whoami)

echo "MoE environment active:"
echo "  venv:       $VIRTUAL_ENV"
echo "  deeptools:  $DEEPTOOLS_INSTALL_DIR"
echo "  torch-spyre: $(python3 -c 'import torch_spyre; print(torch_spyre.__file__)' 2>/dev/null || echo 'not importable')"
echo "  dxp:        $(which dxp_standalone 2>/dev/null || echo 'not found')"
ACTIVATE_EOF
chmod +x "$MOE_ROOT/activate.sh"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate:  source $MOE_ROOT/activate.sh"
echo "To run tests: cd $DTI_PROJECT_ROOT/torch-spyre-moe && python3 -m pytest tests/inductor/test_indirect_access.py -v"
