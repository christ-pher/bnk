#!/bin/sh
# Build the Windows resource objects that give bnk.exe and bnk-tray.exe
# their icon.
#
# Without these the Start menu entry, Alt-Tab, and Explorer all show the
# blank default: an .exe carries its icon as a linked COFF resource, and
# the Go toolchain only picks one up from a .syso in the package
# directory. The objects are committed so the release build needs no
# extra tool, and they are per-arch because a COFF object names the
# machine it was built for.
#
# Run after packaging/icons/gen.py changes bnk.ico:
#     sh packaging/icons/gen-syso.sh
set -eu

root=$(cd "$(dirname "$0")/../.." && pwd)
ico="$root/packaging/windows/bnk.ico"
rsrc="github.com/akavel/rsrc@v0.10.2"

for pkg in bnk bnk-tray; do
  for arch in amd64 arm64; do
    out="$root/cmd/$pkg/rsrc_windows_$arch.syso"
    go run "$rsrc" -ico "$ico" -arch "$arch" -o "$out"
    echo "cmd/$pkg/rsrc_windows_$arch.syso"
  done
done
