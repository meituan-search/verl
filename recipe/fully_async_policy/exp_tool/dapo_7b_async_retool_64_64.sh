set -x

export VLLM_USE_V1=1

# ================= data/model/tool =================
HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

dapo_math_17k=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/wangshulin02/data/BytedTsinghua-SIA/DAPO-Math-17k
aime_2024=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/wangshulin02/data/Maxwell-Jia/AIME_2024
aime_2025=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/wangshulin02/data/yentinglin/aime_2025
model_path=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/wangshulin02/model/checkpoint/multiturn-sft-qwen-2.5-7b-instruct/global_step_372-merged_hf_model
train_files="['$dapo_math_17k']"
test_files="['$aime_2025', '$aime_2024']"

# tool
tool_config_path=recipe/fully_async_policy/exp_tool/sandbox_fusion_tool_config.yaml
retool_path=recipe/retool/retool.py

# wandb / tensorboard
project_name=retool
experiment_name=qwen2.5-7b_dapo_async_tool_64_64_mbs32_tfs4_reqbatch4
default_local_dir=$DATA_ROOT/checkpoint/$experiment_name

# ================= algorithm =================
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns=16
max_prompt_length=2048
max_response_length=16384
actor_lr=1e-6

# ================= perfomance =================
infer_tp=4 # vllm
train_sp=4 # train
fsdp_size=8 # train
offload=True

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))

# ================= async policy =================
rollout_name="vllm"
rollout_mode="async"

NNODES_ROLLOUT=${NNODES_ROLLOUT:-8}
NNODES_TRAIN=${NNODES_TRAIN:-8}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

train_batch_size=0
gen_prompt_bsz=1
n_resp_per_prompt=16
n_resp_per_prompt_val=30
ppo_mini_batch_size=32
total_rollout_steps=$(((512*400)))
test_freq=20
staleness_threshold=0.5
trigger_parameter_sync_step=4
require_batches=4
partial_rollout=True

python3 -m recipe.fully_async_policy.fully_async_main \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=$retool_path \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=$retool_path \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.val_before_train=True \
    trainer.log_val_generations=20 \
    trainer.save_freq=-1 \
    trainer.default_local_dir=$default_local_dir \
    data.gen_batch_size=${gen_prompt_bsz} \
    trainer.nnodes=$NNODES_TRAIN \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    rollout.nnodes=$NNODES_ROLLOUT \
    rollout.n_gpus_per_node=$NGPUS_PER_NODE \
    rollout.total_rollout_steps=$total_rollout_steps \
    rollout.total_epochs=10 \
    rollout.test_freq=$test_freq \
    async_training.staleness_threshold=$staleness_threshold \
    async_training.trigger_parameter_sync_step=$trigger_parameter_sync_step \
    async_training.require_batches=$require_batches \
    async_training.partial_rollout=$partial_rollout \
    async_training.use_rollout_log_probs=True