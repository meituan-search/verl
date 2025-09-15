set -x

train_files=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/data/full_hh_rlhf/rl/train.parquet
test_files=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/data/full_hh_rlhf/rl/train.parquet

HF_MODEL=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/.friday/models/basemodel/Qwen/Qwen3-8B

python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer'\
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=128 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${HF_MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    critic.optim.lr=1e-5 \
    critic.model.path=${HF_MODEL} \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.megatron.param_offload=True \
    critic.megatron.grad_offload=True \
    critic.megatron.optimizer_offload=True \
    critic.megatron.tensor_model_parallel_size=4 \
    reward_model.enable=True \
    reward_model.megatron.param_offload=True \
    reward_model.megatron.tensor_model_parallel_size=4 \
    reward_model.micro_batch_size_per_gpu=4 \
    reward_model.model.path=${HF_MODEL} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='verl_megatron_full_hh_rlhf_examples' \
    trainer.experiment_name='deepseek_llm_7b_model_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=100 $@
