#!/usr/bin/env bash
# Builds the C shim over libtorch.
#
#   ./build.sh             -> ext/libtch.so (self-contained: links libtorch
#                             dynamically and carries an rpath to it)
#   STATIC=1 ./build.sh    -> ext/libtch.a (static shim; the final binary
#                             must then link -ltorch -ltorch_cpu -lc10
#                             -lstdc++ AFTER -ltch)
#
# Environment:
#   LIBTORCH   torch package dir (else autodetected, see bin/detect_torch.sh)
#   CXX        compiler (default g++)
#   CXX11_ABI  override -D_GLIBCXX_USE_CXX11_ABI (default: match torch wheel)
set -euo pipefail
cd "$(dirname "$0")"

TORCH="$(../bin/detect_torch.sh)"
INC=("$TORCH/include" "$TORCH/include/torch/csrc/api/include")
LIB="$TORCH/lib"
CXX="${CXX:-g++}"

# Match the ABI of the installed torch to avoid broken std::string symbols.
if [ -z "${CXX11_ABI:-}" ]; then
  ABI_SRC="$(dirname "$TORCH")"
  # torch package dir -> site-packages/torch; python sits three levels up
  PY="$(cd "$ABI_SRC/.." && pwd)/bin/python"
  [ -x "$PY" ] || PY=python3
  CXX11_ABI="$("$PY" -c 'import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))' 2>/dev/null || echo 1)"
fi

FLAGS=(-std=c++20 -O2 -fPIC -D_GLIBCXX_USE_CXX11_ABI="$CXX11_ABI" -I"${INC[0]}" -I"${INC[1]}")

echo "torch:    $TORCH"
echo "cxx:      $CXX ${FLAGS[*]}"

if [ -n "${STATIC:-}" ]; then
  "$CXX" "${FLAGS[@]}" -c -o torch_api.o torch_api.cpp
  ar rcs libtch.a torch_api.o
  rm -f torch_api.o
  echo "built:    libtch.a (static shim; link the final binary against -ltorch -ltorch_cpu -lc10 -lstdc++ after -ltch)"
else
  CUDA_LIBS=()
  [ -f "$LIB/libtorch_cuda.so" ] && CUDA_LIBS+=(-ltorch_cuda -lc10_cuda)
  "$CXX" "${FLAGS[@]}" -shared -o libtch.so torch_api.cpp \
    -L"$LIB" -ltorch -ltorch_cpu -lc10 "${CUDA_LIBS[@]}" \
    -Wl,-rpath,"$LIB"
  echo "built:    libtch.so"
fi
