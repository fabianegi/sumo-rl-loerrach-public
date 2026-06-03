#!/usr/bin/env bash
# Build corridor network from node/edge definitions using netconvert.
# Idempotent - safe to re-run. Requires SUMO tools (netconvert).
#
# Usage: ./scripts/build_corridor_net.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUMO_DIR="$PROJECT_ROOT/data/sumo_config"

NOD_FILE="$SUMO_DIR/loerrach_corridor.nod.xml"
EDG_FILE="$SUMO_DIR/loerrach_corridor.edg.xml"
OUT_FILE="$SUMO_DIR/loerrach_corridor.net.xml"

# Check netconvert is available
if ! command -v netconvert &>/dev/null; then
    echo "ERROR: netconvert not found. Install SUMO tools or set SUMO_HOME."
    echo "  macOS:   brew install sumo"
    echo "  Linux:   sudo apt install sumo-tools"
    echo "  Windows: included in SUMO installer"
    exit 1
fi

# Check input files
for f in "$NOD_FILE" "$EDG_FILE"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing input file: $f"
        exit 1
    fi
done

echo "Building corridor network..."
echo "  Nodes: $NOD_FILE"
echo "  Edges: $EDG_FILE"
echo "  Output: $OUT_FILE"

netconvert \
    --node-files="$NOD_FILE" \
    --edge-files="$EDG_FILE" \
    --output-file="$OUT_FILE" \
    --no-turnarounds true \
    --junctions.join false

echo "Corridor network built: $OUT_FILE"
echo "  TLS IDs (should be C, J_N, J_S):"
grep -oP 'tlLogic id="\K[^"]+' "$OUT_FILE" 2>/dev/null || echo "  (check manually with: grep tlLogic $OUT_FILE)"
