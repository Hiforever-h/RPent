.. _libero-data-flywheel:

采集轨迹与数据飞轮
========================

使用 ``api`` 规划器时，可通过 ``--record-trajectory`` 保存模型请求、回复和工具调用。已支持动作采集的环境还会记录实际执行的动作及观测，供后续导出为 LeRobot 训练数据。

安装
----

以下示例使用 LIBERO-PRO。环境准备和仿真资源下载见 :doc:`../get_started/quickstart`，API 认证配置见 :doc:`configure_planner`。在 RPent 仓库根目录安装 LIBERO-PRO 及可选的导出依赖：

.. code-block:: bash

   uv pip install -e ".[libero-pro,flywheel]"

采集轨迹
--------

采集默认关闭，且仅在 ``--planner api`` 下生效。动作采集支持 LIBERO、RoboCasa365、RoboTwin 的评测模式，以及 RoboDojo；前三个环境的探索模式只记录规划器和工具调用。开启采集并指定运行目录：

.. code-block:: bash

   rpent --robot libero \
     --suite libero_goal --task 0 --seed 0 \
     --planner api --model anthropic:claude-opus-4-8 \
     --record-trajectory \
     --output-dir /path/to/run

运行结束后，记录保存在 ``/path/to/run/trajectories/<task_id>/`` 下，包括 ``task.json``、``planners.jsonl``、``actions.jsonl``、``predictions.jsonl`` 和存放附件的 ``artifacts/`` 目录。VLA 记录将每次指令与实际执行的动作关联，不保存未执行的预测动作。

导出训练数据
------------

将同一机器人的轨迹导出到新目录：

.. code-block:: bash

   rpent-flywheel \
     --input-root /path/to/run \
     --output-dir /path/to/lerobot/goal-task-00

``--input-root`` 可以指向包含 ``task.json`` 的单条轨迹目录，也可以指向包含多次运行的上级目录。不同机器人应分别导出；``--output-dir`` 必须尚不存在。

只有环境确认成功的任务才参与导出，规划器结束不代表任务成功。导出程序只选取具有所需观测的连续 VLA 动作片段，脚本动作不参与导出。

数据集保存在 ``/path/to/lerobot/goal-task-00/train/``，不会自动划分验证集。输出目录中的 ``report.json`` 记录导出数量和排除原因；没有符合条件的动作时，只生成报告。原始记录不会被修改。

使用 RLinf 训练
---------------

对于上述 LIBERO 示例，可使用导出的 LeRobot 数据集，在独立的 RLinf 环境中进行 Pi0.5 监督微调。RPent 负责采集和导出；模型训练、日志和权重保存由 RLinf 管理。

环境安装和训练配置请参考 `RLinf OpenPI 监督微调指南 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/sft_openpi.html>`_。训练数据路径应指向导出的 LeRobot 数据集目录，即包含 ``meta/info.json`` 的目录，而不是原始轨迹目录。

先准备支持 LIBERO LeRobot 数据加载的 RLinf 环境，并将 Pi0.5 LIBERO SFT 配置保存为 ``/path/to/sft-config/libero_pi05_sft.yaml``，再运行：

.. code-block:: bash

   cd /path/to/RLinf
   unset PYTHONPATH
   export PYTHONPATH="$PWD"
   export EMBODIED_PATH="$PWD/examples/sft"

   .venv/bin/python examples/sft/train_vla_sft.py \
     --config-path /path/to/sft-config \
     --config-name libero_pi05_sft \
     data.train_data_paths=/path/to/lerobot/goal-task-00/train \
     actor.model.model_path=/path/to/pi05-checkpoint \
     runner.logger.log_path=/path/to/new-training-output

根据可用 GPU 设置配置中的资源分配和 batch size，并使用与所选模型和数据集相匹配的 checkpoint 及归一化设置。
