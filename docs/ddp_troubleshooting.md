# DDP Troubleshooting Guide

## Common Error: Inplace Operation During Backward

### Error Message
```
RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation: [torch.cuda.FloatTensor [256]] is at
version 3; expected version 2 instead.
```

### Root Cause
This error occurs in DDP when:
1. **Unused parameters exist** - Some model parameters don't receive gradients
2. **Inplace operations** - Operations like `+=`, `*=` modify tensors in-place
3. **Parameter sharing** - Same parameters used in multiple forward passes

### Solution Applied ✅

**File**: `scripts/train.py`

Changed DDP initialization:
```python
# Before
model = DDP(model, ..., find_unused_parameters=False)

# After
model = DDP(model, ..., find_unused_parameters=True)
```

**Why this works**:
- DDP normally expects all parameters to receive gradients
- Setting `find_unused_parameters=True` tells DDP to handle cases where some parameters don't get gradients
- Slightly slower (~5-10%), but more robust

### Alternative Solutions

#### 1. Find and Fix Unused Parameters

Check which parameters are unused:
```python
# Add after model creation
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"Unused parameter: {name}")
```

Then either:
- Remove unused layers
- Ensure all model branches are used
- Set `requires_grad=False` for unused params

#### 2. Check for Inplace Operations

Search for these patterns in your code:
```python
# Bad - inplace operations
x += y
x.add_(y)
x *= 2
F.relu(x, inplace=True)

# Good - out-of-place operations
x = x + y
x = x.add(y)
x = x * 2
x = F.relu(x)
```

#### 3. Enable Anomaly Detection (Debug Mode)

Add to your training script to find the exact operation:
```python
import torch
torch.autograd.set_detect_anomaly(True)

# Your training code
# Will show which operation caused the error
```

**Note**: Only use for debugging - makes training ~10x slower!

### Performance Impact

| Setting | Speed | Robustness |
|---------|-------|------------|
| `find_unused_parameters=False` | Fastest | Can break |
| `find_unused_parameters=True` | ~5-10% slower | Robust ✅ |

### When to Use Each Setting

**Use `find_unused_parameters=False`** when:
- All model parameters always receive gradients
- No conditional branches in forward pass
- Maximum performance needed
- Model is well-tested

**Use `find_unused_parameters=True`** when:
- Model has conditional branches (if/else in forward)
- Some features can be disabled (use_proprio, use_dino)
- Getting inplace operation errors
- Developing/debugging

### Other Common DDP Errors

#### NCCL Timeout
```
RuntimeError: NCCL error: unhandled system error
```

**Solution**:
```bash
export NCCL_TIMEOUT=3600  # Increase timeout
export NCCL_DEBUG=INFO     # Enable debug logs
```

#### Port Already in Use
```
DistNetworkError: address already in use
```

**Solution**:
```bash
# Use different port
torchrun --master_port=29502 scripts/train.py ...
```

#### Out of Memory in DDP
```
CUDA out of memory
```

**Solution**:
```yaml
# Reduce batch size per GPU
data:
  batch_size: 512  # Instead of 2048
  # Effective batch: 512 × num_gpus
```

### Testing Your Fix

Run a quick test:
```bash
# Single GPU (should work)
python scripts/train.py --config configs/experiments/multi_task_327_352_356.yaml

# Multi-GPU (test DDP fix)
torchrun --nproc_per_node=2 --master_port=29501 \
    scripts/train.py \
    --config configs/experiments/multi_task_327_352_356.yaml
```

If it trains for 1-2 batches without error, the fix worked!

### Monitoring DDP Performance

Check if DDP is working efficiently:
```python
# In training loop
import torch.distributed as dist

if dist.is_initialized():
    # Time communication overhead
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    loss.backward()  # Triggers all-reduce
    end.record()
    torch.cuda.synchronize()

    print(f"Backward pass time: {start.elapsed_time(end):.2f}ms")
```

**Healthy values**:
- Small models: 10-50ms
- Large models: 50-200ms
- If >500ms: Check network or reduce model size

### Best Practices

1. **Always test single GPU first**
   ```bash
   python scripts/train.py ...  # Test without DDP
   ```

2. **Start with 2 GPUs**
   ```bash
   torchrun --nproc_per_node=2 ...  # Test DDP
   ```

3. **Scale up gradually**
   ```bash
   torchrun --nproc_per_node=4 ...  # Then 4, 8, etc.
   ```

4. **Monitor GPU utilization**
   ```bash
   watch -n 1 nvidia-smi  # Check GPU usage
   ```

5. **Check for stragglers**
   - All GPUs should have ~same utilization
   - If one GPU is idle → data imbalance

### Summary

✅ **Current Fix**: `find_unused_parameters=True` enabled
- Slightly slower but robust
- Handles conditional model paths
- Prevents inplace operation errors

🎯 **For Best Performance**:
- Profile to find unused parameters
- Remove or fix them
- Switch back to `find_unused_parameters=False`

📊 **Expected Overhead**: 5-10% slower with `find_unused_parameters=True`
