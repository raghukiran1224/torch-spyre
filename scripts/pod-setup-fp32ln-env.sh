#!/bin/bash
# =============================================================================
# FP32 LayerNorm Isolated Environment Setup for Spyre Pod
# =============================================================================
# Creates a fully isolated venv for testing fp32 layernorm with dtype casting
# from PR #1691. Uses system deeptools (master has dl16tofp32/fp32todl16).
#
# Usage:
#   bash pod-setup-fp32ln-env.sh          # Full setup (first time)
#   bash pod-setup-fp32ln-env.sh --quick  # Just reinstall torch-spyre
#
# After setup, activate with:
#   source $DTI_PROJECT_ROOT/fp32ln-env/activate.sh
# =============================================================================
set -e

FP32LN_ROOT="$DTI_PROJECT_ROOT/fp32ln-env"
FP32LN_VENV="$FP32LN_ROOT/venv"
QUICK=0

for arg in "$@"; do
    case $arg in
        --quick) QUICK=1 ;;
    esac
done

echo "=== FP32 LayerNorm Environment Setup ==="
echo "  Root:       $FP32LN_ROOT"
echo "  Venv:       $FP32LN_VENV"
echo "  Quick mode: $QUICK"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 1. Ensure worktree exists and is up-to-date
# ─────────────────────────────────────────────────────────────────────────────
echo "[1/4] Updating source worktree..."

if [ ! -d "$DTI_PROJECT_ROOT/torch-spyre-fp32ln" ]; then
    cd "$DTI_PROJECT_ROOT/torch-spyre"
    git remote add fork https://github.com/raghukiran1224/torch-spyre.git 2>/dev/null || true
    git fetch fork fp32-layernorm-test
    git worktree add "$DTI_PROJECT_ROOT/torch-spyre-fp32ln" fork/fp32-layernorm-test
else
    cd "$DTI_PROJECT_ROOT/torch-spyre-fp32ln"
    git fetch fork fp32-layernorm-test 2>/dev/null || git fetch origin fp32-layernorm-test 2>/dev/null || true
    git checkout --detach FETCH_HEAD 2>/dev/null || true
fi
echo "  torch-spyre-fp32ln: $(cd $DTI_PROJECT_ROOT/torch-spyre-fp32ln && git log --oneline -1)"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Create isolated venv
# ─────────────────────────────────────────────────────────────────────────────
echo "[2/4] Setting up isolated venv..."
mkdir -p "$FP32LN_ROOT"

if [ ! -d "$FP32LN_VENV" ]; then
    python3 -m venv "$FP32LN_VENV" --system-site-packages
    echo "  Created new venv at $FP32LN_VENV"
else
    echo "  Venv already exists at $FP32LN_VENV"
fi

source "$FP32LN_VENV/bin/activate"
pip install pytest --quiet 2>/dev/null

# ─────────────────────────────────────────────────────────────────────────────
# 3. Install torch-spyre from fp32ln branch
# ─────────────────────────────────────────────────────────────────────────────
echo "[3/4] Installing torch-spyre (fp32-layernorm-test branch)..."
cd "$DTI_PROJECT_ROOT/torch-spyre-fp32ln"
pip install -e . --no-deps --no-build-isolation > /dev/null 2>&1
echo "  torch-spyre installed from $DTI_PROJECT_ROOT/torch-spyre-fp32ln"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Create activation script
# ─────────────────────────────────────────────────────────────────────────────
echo "[4/4] Creating activation script..."
cat > "$FP32LN_ROOT/activate.sh" << 'ACTIVATE_EOF'
#!/bin/bash
# Source this to activate the FP32 LayerNorm isolated environment.
# Usage: source $DTI_PROJECT_ROOT/fp32ln-env/activate.sh

_FP32LN_ROOT="$DTI_PROJECT_ROOT/fp32ln-env"

# Activate venv
source "$_FP32LN_ROOT/venv/bin/activate"

# Use system deeptools (master has dl16tofp32/fp32todl16)
# No custom deeptools build needed.

# Clean inductor cache (stale artifacts cause confusing failures)
rm -rf /tmp/torchinductor_$(whoami)

echo "FP32 LayerNorm environment active:"
echo "  venv:        $VIRTUAL_ENV"
echo "  deeptools:   ${DEEPTOOLS_INSTALL_DIR:-system}"
echo "  torch-spyre: $(python3 -c 'import torch_spyre; print(torch_spyre.__file__)' 2>/dev/null || echo 'not importable')"
echo "  dxp:         $(which dxp_standalone 2>/dev/null || echo 'not found')"
ACTIVATE_EOF
chmod +x "$FP32LN_ROOT/activate.sh"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate:  source $FP32LN_ROOT/activate.sh"
echo ""
echo "To test fp32 layernorm:"
echo "  TORCH_SPYRE_FP32_LAYERNORM=1 python3 -c \""
echo "  import torch"
echo "  def ln_fp32(x):"
echo "      return torch.nn.functional.layer_norm(x, [4096])"
echo "  compiled = torch.compile(ln_fp32)"
echo "  x = torch.randn(32, 4096, dtype=torch.float16, device='spyre')"
echo "  result = compiled(x)"
echo "  print(result.shape, result.dtype)"
echo "  \""
