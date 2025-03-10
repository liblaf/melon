#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

UPSTREAM_DIR="$HOME/net/SeaDrive/My Libraries/dataset/2024-09-13-template-head/sculptor/"
DATA_DIR="./data/00-raw"
mkdir --parents --verbose "$DATA_DIR"
cp --archive --force --verbose "$UPSTREAM_DIR"/{face,cranium,mandible}.ply "$DATA_DIR/"
