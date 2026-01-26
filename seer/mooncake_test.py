from mooncake.store import MooncakeStoreClient
from mooncake.transfer import TransferEngine

# 1. 初始化 Mooncake 客户端（对接全局 KVCache 池）
store_client = MooncakeStoreClient(etcd_endpoints=["http://etcd-node-01:2379"])
# 2. 初始化传输引擎（RDMA 加速迁移）
te = TransferEngine(rdma_device="roce", num_channels=2)

# 3. 模拟 verl KV 接口测试（key 格式需与 verl 0.7 兼容）
kv_key = "rollout/group_001/request_001/chunk_001"  # 符合 verl 请求ID规范
kv_value = {
    "key": torch.randn(1, 1024, 4096).cuda(),  # KVCache Key 张量（verl 0.7 要求格式）
    "value": torch.randn(1, 1024, 4096).cuda(), # KVCache Value 张量
    "metadata": {"group_id": "group_001", "chunk_idx": 0}  # 自定义元数据
}
# 写入全局缓存
store_client.put(kv_key, kv_value, ttl=3600)  # TTL 避免缓存泄露
# 读取全局缓存（跨实例复用）
cached_kv = store_client.get(kv_key)
assert cached_kv is not None, "Mooncake 与 verl 0.7 KV 接口不兼容"