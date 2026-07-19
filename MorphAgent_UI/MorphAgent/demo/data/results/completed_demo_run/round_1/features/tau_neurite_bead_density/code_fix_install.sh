#!/usr/bin/env bash
set -e
# Execute the install command in conda environment morphagent
conda run -n morphagent bash << 'INSTALL_EOF'
pip install numpy scipy scikit-image
INSTALL_EOF

