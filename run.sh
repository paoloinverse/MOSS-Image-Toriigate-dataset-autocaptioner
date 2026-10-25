#!/usr/bin/env bash
# Launch the MOSS Image Auto-Captioner (Ubuntu/Xubuntu, X11 or Wayland).
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
exec python3 -m app.main "$@"
