# Check cluster partitions and available nodes
sinfo -p gpu

# Check current queue and running jobs  
squeue -p gpu

# Check specific GPU node configuration
scontrol show partition gpu

# Check available resources on GPU nodes
sinfo -p gpu -o "%N %c %m %G %a %T"
