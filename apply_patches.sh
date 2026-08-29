#!/bin/bash
# Apply third-party patches after pip install

# Fix robustness (torchvision.models.utils removed in newer torchvision)
ROBUSTNESS_DIR=$(python -c "import robustness; import os; print(os.path.dirname(robustness.__file__))")
find "$ROBUSTNESS_DIR" -name "*.py" -exec sed -i 's/from torchvision.models.utils import load_state_dict_from_url/from torch.hub import load_state_dict_from_url/g' {} +
echo "Patched robustness"