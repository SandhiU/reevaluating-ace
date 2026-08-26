import os, sys, torch
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
print('TF32 OFF | cudnn:', torch.backends.cudnn.allow_tf32, '| matmul:', torch.backends.cuda.matmul.allow_tf32)
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
import runpy
runpy.run_path(script, run_name='__main__')
