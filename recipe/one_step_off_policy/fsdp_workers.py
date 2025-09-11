# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoConfig

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer, omega_conf_to_dataclass
from verl.utils.debug import DistProfiler, DistProfilerExtension, log_gpu_memory_usage
from verl.utils.device import (
    get_device_name,
    get_nccl_backend,
    get_torch_device,
)
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    fsdp_version,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import get_generation_config, update_model_config
from verl.utils.vllm_utils import patch_vllm_moe_model_weight_loader
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

__all__ = ["DetachActorWorker", "DetachRolloutWorker", "DetachAsyncRolloutWorker", "CriticWorker"]


def get_inference_model(rollout):
    """
    根据不同类型的inference_engine获取模型对象
    Args:
        rollout: rollout对象，包含inference_engine
    Returns:
        model: 模型对象
    """
    inference_engine = getattr(rollout, "inference_engine", None)
    print(f"[get_inference_model] Debug: inference_engine = {inference_engine}")
    if inference_engine is None:
        # Engine might be deferred-initialized (e.g., Async SGLang/vLLM). Skip for now.
        print(f"[get_inference_model] Debug: inference_engine is None, returning None")
        return None
    
    # 判断inference_engine的类型
    print(f"[get_inference_model] Debug: inference_engine type = {type(inference_engine)}")
    print(f"[get_inference_model] Debug: has llm_engine = {hasattr(inference_engine, 'llm_engine')}")
    print(f"[get_inference_model] Debug: has worker = {hasattr(inference_engine, 'worker')}")
    print(f"[get_inference_model] Debug: has model_executor = {hasattr(inference_engine, 'model_executor')}")
    print(f"[get_inference_model] Debug: has model = {hasattr(inference_engine, 'model')}")
    
    if hasattr(inference_engine, "llm_engine"):
        # LLM类型 - vLLMRollout
        print(f"[get_inference_model] Debug: Using vLLMRollout path")
        inference_model = inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
    elif hasattr(inference_engine, "worker"):
        # WorkerWrapperBase类型 - vLLMAsyncRollout
        print(f"[get_inference_model] Debug: Using vLLMAsyncRollout path")
        inference_model = inference_engine.worker.model_runner.model
    elif hasattr(inference_engine, "model_executor"):
        # AsyncEngine类型 - SGLangAsyncRollout
        print(f"[get_inference_model] Debug: Using SGLangAsyncRollout path")
        inference_model = inference_engine.model_executor.driver_worker.worker.model_runner.model
    elif hasattr(inference_engine, "model"):
        # SGLang Engine类型 - SGLangRollout (同步)
        print(f"[get_inference_model] Debug: Using SGLangRollout path")
        inference_model = inference_engine.model
    else:
        print(f"[get_inference_model] Debug: No matching attributes found, returning None")
        return None
    print(f"[get_inference_model] Debug: Final inference_model = {inference_model}")
    return inference_model


class DetachNcclSync(ActorRolloutRefWorker):
    def _get_actor_params(self):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self):
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None

        print(f"[DetachNcclSync] Debug: rollout_name = {self.config.rollout.name}")
        params = self._get_actor_params() if self._is_actor else None
        if self._is_rollout:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else "N/A"
            print(f"[DetachNcclSync] Debug: rank={rank}, rollout.inference_engine = {getattr(self.rollout, 'inference_engine', 'NOT_FOUND')}")
            inference_model = get_inference_model(self.rollout)
            print(f"[DetachNcclSync] Debug: get_inference_model returned = {inference_model}")
            if inference_model is None:
                # 在分布式 SGLang 中，只有主进程有 inference_engine
                # 非主进程应该跳过权重同步，但继续执行广播逻辑
                print(f"[DetachNcclSync] Engine not ready; skip weight loading but continue with broadcast (rank={rank})")
                # 不要 return，继续执行广播逻辑
            
            # 只有在有 inference_model 时才进行权重加载相关的处理
            if inference_model is not None:
                rollout_name = self.config.rollout.name
                if rollout_name == "vllm":
                    patch_vllm_moe_model_weight_loader(inference_model)
                elif rollout_name == "sglang":
                    # SGLang 不需要 patch
                    pass
                else:
                    print(f"[DetachNcclSync] Warning: Unknown rollout name {rollout_name}, skipping weight loader patch")
        for key, shape, dtype in self._weights_info:
            tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
            if self._is_actor:
                assert key in params
                origin_data = params[key]
                if hasattr(origin_data, "full_tensor"):
                    origin_data = origin_data.full_tensor()
                if torch.distributed.get_rank() == 0:
                    tensor.copy_(origin_data)
            from ray.util.collective import collective

            collective.broadcast(tensor, src_rank=0, group_name="actor_rollout")
            if self._is_rollout and inference_model is not None:
                # 根据推理引擎类型使用不同的权重加载方法
                rollout_name = self.config.rollout.name
                if rollout_name == "vllm":
                    inference_model.load_weights([(key, tensor)])
                elif rollout_name == "sglang":
                    inference_model.update_weights_from_tensor([(key, tensor)])
                else:
                    print(f"[DetachNcclSync] Warning: Unknown rollout name {rollout_name}, skipping weight loading for {key}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_weights_info(self):
        assert self._is_actor
        if hasattr(self, "_weights_info"):
            return self._weights_info
        if fsdp_version(self.actor_module_fsdp) == 1:
            from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType

            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )
        params = self._get_actor_params()
        ret = []
        for key, tensor in params.items():
            ret.append((key, tensor.size(), tensor.dtype))
        self._weights_info = ret
        return ret


class DetachActorWorker(DetachNcclSync):
    def _get_actor_params(self):
        assert self._is_actor
        params = self.actor_module_fsdp.state_dict()
        from verl.utils.model import convert_weight_keys

        params = convert_weight_keys(
            params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        )
        return params


class DetachRolloutWorker(DetachNcclSync):
    def __init__(self, config: DictConfig, role: str):
        Worker.__init__(self)
        assert role == "rollout"
        self.config = config
        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
        # TODO(haibin.lin):
        # As of now the type of config is DictConfig, if we assign config.profiler with ProfilerConfig,
        # it will actually convert the ProfilerConfig dataclass back to a DictConfig.
        # We can still use ProfilerConfig for testing purpose (tests/utils/test_nvtx_profile.py)
        # as they provides DictConfig-like interface
        # The benefit of creating the dataclass config is to perform validation during __post_init__
        profiler_config = omega_conf_to_dataclass(config.rollout.get("profiler", {}))
        DistProfilerExtension.__init__(self, DistProfiler(rank=self.rank, config=profiler_config))
        self._is_rollout = True
        self._is_actor = False

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))

        use_shm = self.config.model.get("use_shm", False)
        local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
        trust_remote_code = self.config.model.get("trust_remote_code", False)

        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2"
        )

        # patch for kimi-vl
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        rollout_name = self.config.rollout.name

        from verl.workers.rollout.vllm_rollout import vLLMRollout, vLLMAsyncRollout
        from verl.workers.rollout.sglang_rollout import SGLangRollout, SGLangAsyncRollout

        log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)

        if rollout_name == "vllm":
            vllm_rollout_cls = vLLMRollout if self.config.rollout.mode == "sync" else vLLMAsyncRollout
            rollout = vllm_rollout_cls(
                model_path=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=actor_model_config,
                device_mesh=rollout_device_mesh,
                trust_remote_code=trust_remote_code,
            )
        elif rollout_name == "sglang":
            sglang_rollout_cls = SGLangRollout if self.config.rollout.mode == "sync" else SGLangAsyncRollout
            rollout = sglang_rollout_cls(
                actor_module=local_path,
                config=self.config.rollout,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                model_hf_config=actor_model_config,
                device_mesh=rollout_device_mesh,
                trust_remote_code=trust_remote_code,
            )
        else:
            raise NotImplementedError(f"Rollout name: {rollout_name} is not supported")

        log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)

        from .detach_sharding_manager import DetachShardingManager

        sharding_manager = DetachShardingManager(
            inference_engine=rollout.inference_engine, device_mesh=rollout_device_mesh
        )

        log_gpu_memory_usage("After building sharding manager", logger=logger)

        self.rollout = rollout
        self.rollout_sharding_manager = sharding_manager

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
    def async_generate_sequences(self, *args, **kwargs):
        return super().generate_sequences(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, weights_info):
        assert self._is_rollout
        self._weights_info = weights_info


class DetachAsyncRolloutWorker(AsyncActorRolloutRefWorker, DetachRolloutWorker):
    def __init__(self, config: DictConfig, role: str):
        print(f"[DetachAsyncRolloutWorker] {DetachAsyncRolloutWorker.__mro__}")
        DetachRolloutWorker.__init__(self, config, role)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        print("[DetachAsyncRolloutWorker] init_model")
        DetachRolloutWorker.init_model(self)

        self.vllm_tp_size = self.config.rollout.tensor_model_parallel_size
        self.vllm_dp_rank = int(os.environ["RANK"]) // self.vllm_tp_size
        self.vllm_tp_rank = int(os.environ["RANK"]) % self.vllm_tp_size

        # used for sleep/wake_up
        self.rollout.sharding_manager = self.rollout_sharding_manager
