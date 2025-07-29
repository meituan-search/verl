# 基于verl的改造方案

架构图

![full_async](
https://raw.githubusercontent.com/ArronHZG/verl-community/205b491d169ac026261c1433cfe8e8696dc46fab/docs/full_async.svg)

方案1参考StreamRL

![StreamRL](
https://raw.githubusercontent.com/ArronHZG/verl-community/205b491d169ac026261c1433cfe8e8696dc46fab/docs/StreamRL.png)

方案2参考Mistralai

![mistralai](
https://raw.githubusercontent.com/ArronHZG/verl-community/205b491d169ac026261c1433cfe8e8696dc46fab/docs/mistralai.png)


为实现纯异步训练工作流，基于已有的 one step off policy 代码 扩增实现Rollouter 以及 Message Queue，对Trainer进行更新。

整体的训练流程参考StreamRL，将原有流程中生成 train_batch_size个样本后进行下一步训练的过程，修改为流式的样本传递，train
拿到一次前向的样本后就进行样本分发（ppo_mini_batch_size*worker）。与one-step-off相比，我们将一次step的异步，继续细化到一次batch的异步。

**MessageQueue** 作为Ray的Actor存在，支持zeromq消息队列保存生成的样本，并提供给Trainer使用。Trainer 和 Rollouter 都持有
MessageQueue 的Handler，通过接口完成样本的插入与消费。

**FullyAsyncRollouter** 类似于现有的 Trainer，实现fit()工作流，循环调用 Rollout 进行样本的生成。FullyAsyncRollouter 对于已有的
vLLMAsyncRollout SGLangAsyncRollout 进行封装。

* 方案1，使用异步更新策略，FullyAsyncRollouter 根据样本生成的进展，自动访问PS，判断是否进行新的参数加载。

* 方案2，参考PR https://github.com/volcengine/verl/pull/2246​​https://github.com/volcengine/verl/pull/2200​，Rollout
组件需要支持暂停及恢复，从而进行参数的更新。暂停时，需要保存进行中的rollout样本，下次继续恢复生产。

**FullyAsyncTrainer** 与当前实现类似，区别是样本的获取修改为从Queue中获取，Queue有最少batch样本就开始进行分发。rainer完成一次step的训练后，
与FullyAsyncRollouter的使用策略对应：

* 方案1，使用异步更新策略，参数产生后，主动同步到PS中。

* 方案2，直接调用Rollouter进行同步，主动通知Rollouter暂停生成，进行参数的同步更新。

基于以上的思路，

当Rollouter生产快于Trainer消费时，queue中会存在多步过期的样本，我们需要在Rollouter中设置“新鲜度 staleness
”阈值，由当前的参数版本以及生成的样本数量，决定是否要暂停生成。zeromq 的最大长度应为 freshness * total_size，并且实现基于新鲜度的拒绝策略。

当使用方案1时，参数的同步会更加及时，新鲜度低的样本数量较多，但是长尾样本由不同的参数产生，长尾样本的不同token所对应的参数版本会传递给训练引擎，后续可能可以根据这一信息对loss进行加权处理。

当使用方案2时，参数的同步由FullyAsyncRollouter主动控制，触发时机取决长尾样生成完成以及参数已就绪，长尾的样本所使用的参数版本一致，但是会有"
rollout空洞"产生。

当Rollouter生产慢于Trainer消费时，基本等价于同步训练，区别是最后一批样本是否丢弃或者重播。