.. _libero-data-flywheel:

Data Flywheel
=============

Use ``--record-trajectory`` with the ``api`` planner to save model requests,
responses, and tool calls. Supported environments also record executed actions
and their observations for later export to LeRobot training data.

Installation
------------

The example below uses LIBERO-PRO. Follow :doc:`../get_started/quickstart` for
environment setup and simulator assets, and :doc:`configure_planner` for API
credentials. From the RPent repository root, install LIBERO-PRO with the
optional export dependencies:

.. code-block:: bash

   uv pip install -e ".[libero-pro,flywheel]"

Record a Trajectory
-------------------

Recording is off by default and requires ``--planner api``. Action recording
supports LIBERO, RoboCasa365, and RoboTwin in evaluation mode, and RoboDojo.
Exploration runs in the first three environments record planner and tool calls
only. Enable recording and choose a run directory:

.. code-block:: bash

   rpent --robot libero \
     --suite libero_goal --task 0 --seed 0 \
     --planner api --model anthropic:claude-opus-4-8 \
     --record-trajectory \
     --output-dir /path/to/run

After the run, records are saved under ``/path/to/run/trajectories/<task_id>/``:
``task.json``, ``planners.jsonl``, ``actions.jsonl``, ``predictions.jsonl``, and
an ``artifacts/`` directory for attachments. VLA records link each instruction
to the actions actually executed; predicted actions that were not executed are
not saved.

Export Training Data
--------------------

Export recorded trajectories from one robot to a new output directory:

.. code-block:: bash

   rpent-flywheel \
     --input-root /path/to/run \
     --output-dir /path/to/lerobot/goal-task-00

``--input-root`` accepts a trajectory directory containing ``task.json`` or a
parent directory containing multiple runs. Export each robot separately;
``--output-dir`` must not already exist.

Only tasks confirmed successful by the environment are eligible; planner
completion alone is insufficient. The exporter selects only continuous segments
of executed VLA actions with the required observations; scripted actions are
excluded.

The dataset is written to ``/path/to/lerobot/goal-task-00/train/`` without an
automatic validation split. ``report.json`` in the output directory lists
export counts and exclusions. If no actions qualify, only the report is
written. Original recordings are left unchanged.

Train with RLinf
----------------

For the LIBERO example, use the exported LeRobot dataset for Pi0.5 supervised
fine-tuning in a separate RLinf environment. RPent handles collection and
export; RLinf handles model training, logging, and checkpoint saving.

For environment setup and training configuration, see the
`RLinf OpenPI SFT guide <https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/sft_openpi.html>`_.
Point the training data path to the exported LeRobot dataset directory containing
``meta/info.json``, not the raw episode directory.

Use an RLinf environment with LIBERO LeRobot data-loading support. Prepare a
Pi0.5 LIBERO SFT configuration at
``/path/to/sft-config/libero_pi05_sft.yaml``, then run:

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

Match the configuration's resource placement and batch size to your available
GPUs, and use the checkpoint and normalization settings for the selected model
and dataset.
