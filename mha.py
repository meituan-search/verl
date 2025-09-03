import ray
import torch
import torch.nn as nn

torch.manual_seed(123)
print(f"PyTorch version: {torch.__version__}")

batch_size = 1
context_len = 256
embed_dim = 256


def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time

        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print("%s cost time: %.5f s" % (func.__name__, time_spend))
        return result

    return func_wrapper


class MHAPyTorchScaledDotProduct(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0.0 if not self.training else self.dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True
        )

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        context_vec = self.proj(context_vec)

        return context_vec


@ray.remote(num_gpus=1)
@timer
def run_mha_on_gpu():
    """在指定GPU上运行MHA测试"""
    num_gpus = torch.cuda.device_count()
    print(f"num_gpus: {num_gpus}")

    device = torch.device("cuda")
    embeddings = torch.randn((batch_size, context_len, embed_dim), device=device)
    mha_pytorch_scaled = MHAPyTorchScaledDotProduct(
        d_in=embed_dim, d_out=embed_dim, context_length=context_len, dropout=0.0, num_heads=8, qkv_bias=False
    ).to(device)

    # 执行前向传播
    while True:
        out = mha_pytorch_scaled(embeddings)

    return {
        "output_shape": out.shape,
        "output_mean": out.mean().item(),
        "memory_allocated": torch.cuda.memory_allocated(device) / 1024**3,  # GB
        "memory_reserved": torch.cuda.memory_reserved(device) / 1024**3,  # GB
    }


def main():
    # 初始化Ray
    ray.init()

    print("Ray cluster resources:")
    cluster_resources = ray.cluster_resources()
    print(cluster_resources)

    # 动态获取GPU数量
    num_gpus = int(cluster_resources.get("GPU", 0))
    print(f"Available GPUs: {num_gpus}")

    if num_gpus == 0:
        print("No GPUs available in the Ray cluster!")
        return

    # 创建任务，每个任务使用一个GPU
    gpu_tasks = []
    for gpu_id in range(num_gpus):
        task = run_mha_on_gpu.remote()
        gpu_tasks.append(task)

    print(f"Submitted {len(gpu_tasks)} tasks to Ray cluster")

    # 等待所有任务完成并收集结果
    results = ray.get(gpu_tasks)

    # 打印结果
    print("\n" + "=" * 60)
    print("Multi-GPU MHA Execution Results:")
    print("=" * 60)

    for result in results:
        print(f"  Output shape: {result['output_shape']}")
        print(f"  Output mean: {result['output_mean']:.6f}")
        print(f"  Memory allocated: {result['memory_allocated']:.2f} GB")
        print(f"  Memory reserved: {result['memory_reserved']:.2f} GB")
        print()

    # 计算统计信息
    total_memory_allocated = sum(r["memory_allocated"] for r in results)
    total_memory_reserved = sum(r["memory_reserved"] for r in results)
    avg_output_mean = sum(r["output_mean"] for r in results) / len(results)

    print("Summary:")
    print(f"  Total memory allocated: {total_memory_allocated:.2f} GB")
    print(f"  Total memory reserved: {total_memory_reserved:.2f} GB")
    print(f"  Average output mean: {avg_output_mean:.6f}")


if __name__ == "__main__":
    main()

