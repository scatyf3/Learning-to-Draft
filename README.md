# LEARNING TO DRAFT: ADAPTIVE SPECULATIVE DECODING WITH REINFORCEMENT LEARNING

This is the official implementation of "Learning To Draft: Adaptive Speculative Decoding with Reinforcement Learning" (ICLR 2026).  We introduce Learning to Draft (LTD), a novel method that directly
optimizes for throughput of each draft-and-verify cycle in speculative decoding. We formulate the problem
as a reinforcement learning environment and train two co-adaptive policies to
dynamically coordinate the draft and verification phases. This encourages the
policies to adapt to each other and explicitly maximize decoding efficiency. 

<p align="center">
    <img src="pictures/11.png" width="100%"> <br>
</p>

## Highlight Results

During greedy decoding, LTD yields substantial throughput
improvements over the highly optimized Eagle3 baseline: 36.4% on Qwen3-32B, 9.5% on DeepseekDistilled-Llama-8B, 6.5% on Llama-3-8B, 5% on Vicuna-1.3-13B, and 4% on Qwen3-14B. 

<p align="center">
    <img src="pictures/dav.png" width="100%"> <br>
</p>

## Setup & Installation

```python
git clone https://github.com/zhzihao/Learning-to-Draft.git
cd Learning-to-Draft
python3.11 -m venv ~/venvs/ea_env
source ~/venvs/ea_env/bin/activate
pip install -r requirements.txt
```

## Train Policy 

```bash
export CUDA_VISIBLE_DEVICES=$1

# Model Paths
base_model_path="meta-llama/Llama-3.1-8B-Instruct"
ea_model_path="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
rl_token_model_path="" # Leave empty to use default 60 tokens
rl_checkpoint_path=""  # Leave empty to start from scratch

# Data and Save Directories
data_dir="./eagle/data"
dataset_train="humaneval"
save_path="./checkpoints"

# RL Hyperparameters
total_timesteps=1000000 # paper说了是1m step
batch_size=256 # minibatch size of 256
n_steps=2048 # The PPO is configured with a rollout buffer of 2048 steps
lr=1e-3 
gamma=0.999
#  20 epoch, 0.01 entropy coef A3
n_epochs=20
ent_coef=0.01

python3 -m rl.rl_depth \
    --base_model_path ${base_model_path} \
    --ea_model_path ${ea_model_path} \
    --rl_token_model_path "${rl_token_model_path}" \
    --rl_checkpoint_path "${rl_checkpoint_path}" \
    --data_dir ${data_dir} \
    --dataset_train ${dataset_train} \
    --save_path ${save_path} \
    --total_timesteps ${total_timesteps} \
    --batch_size ${batch_size} \
    --n_steps ${n_steps} \
    --lr ${lr} \
    --gamma ${gamma} \
    --n_epochs ${n_epochs} \
    --ent_coef ${ent_coef} \
    --pi_arch 1024 \
    --vf_arch 1024 256
```

Implementation Details We implement the Proximal Policy Optimization (PPO) algorithm using the Stable-Baselines3 library (Raffin et al., 2021) and train our policies on the HumanEval
dataset (Chen et al., 2021). The size and depth policies are trained for 100k and 1M steps, respectively. Further analysis regarding the choice of training steps can be found in the Appendix C.1.
The PPO is configured with a rollout buffer of 2048 steps and a minibatch size of 256. Due to the
hyperight nature of the two policies, the RL training phase is highly efficient, dettailed can be found
in Appendix A.7. A comprehensive list of all hyperparameters can be found in Appendix A.3.

hyper params
1. Size Policy - Model: 2 ffn, 1024-256, relu, gamma=0.9
2. depth policy: 1 1024 layer, with gamma=0.999
3. Shared params: 20 epoch, 0.01 entropy coef, 1% preheat, 1e-3 lr

### 1. Model & Path Configurations

* **`--base_model_path`**: Path to your base Large Language Model (e.g., LLaMA-3). 
* **`--ea_model_path`**: Path to your EAGLE draft model. 
* **`--rl_token_model_path`**: Path to a pre-trained RL token model `.zip` file.
  * *Usage*: If provided, the environment will use this model to dynamically predict the `total_tokens_to_select`. If left empty (`""`), it defaults to a fixed size of 60 tokens.
* **`--rl_checkpoint_path`**: Path to an existing RL checkpoint to resume training.
  * *Usage*: If you want to continue training from a previous state, provide the `.zip` path. If left empty, a fresh PPO model will be initialized.
* **`--data_dir`**: Base directory where your datasets are stored.
* **`--save_path`**: Directory to save the trained models, checkpoints, and TensorBoard logs.

### 2. Dataset Configurations

* **`--dataset_train`**: The specific dataset to use for training (e.g., `humaneval`). The script expects the data to be located at `{data_dir}/{dataset_train}/question.jsonl`.

### 3. PPO Hyperparameters

* **`--total_timesteps`**: Total number of interactions (steps) the policy will have with the environment. Increase this for longer training.
* **`--lr`**: Initial learning rate (paper setting: `1e-3`). The script uses a custom AdaWM schedule (warmup followed by linear decay).
* **`--warmup_timesteps`**: Number of timesteps for the learning rate to warm up from 0 to `lr`.
* **`--n_steps`**: Number of steps to run for each environment per update. Paper uses `2048`.
* **`--batch_size`**: Minibatch size for the optimization step. Paper uses `256`.
* **`--n_epochs`**: Number of epochs when optimizing the surrogate loss. Paper uses `20`.
* **`--gamma`**: Discount factor for future rewards. Paper uses `0.9` for size policy and `0.999` for depth policy.
* **`--ent_coef`**: Entropy coefficient for the loss calculation. Paper uses `0.01`.
  * *Usage*: Increase this (e.g., `0.01`) if you want to encourage the policy to explore more diverse actions; keep it at `0` for pure exploitation.
* **`--eval_freq`**: Frequency (in timesteps) to save model checkpoints and run evaluations.

### 4. Network Architecture

* **`--pi_arch`**: Architecture of the Policy network (actor).
  * *Usage*: Paper settings are `--pi_arch 1024 256` for size policy and `--pi_arch 1024` for depth policy.
* **`--vf_arch`**: Architecture of the Value network (critic). Paper setting is `--vf_arch 1024 256` for both policies.

After modifying parameters, run:

```bash 

sh train_size.sh
# for training size policy
sh train_depth.sh
# for training depth policy

```
You can change **`--rl_token_model_path`** and **`--rl_checkpoint_path`** for iterative training.

## experients feedback

```
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 8.51        |
|    ep_rew_mean          | 1.47        |
| time/                   |             |
|    fps                  | 127         |
|    iterations           | 33          |
|    time_elapsed         | 531         |
|    total_timesteps      | 67584       |
| train/                  |             |
|    approx_kl            | 0.026262213 |
|    clip_fraction        | 0.0704      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.165      |
|    explained_variance   | 0.381       |
|    learning_rate        | 0.000944    |
|    loss                 | 0.116       |
|    n_updates            | 640         |
|    policy_gradient_loss | 0.00323     |
|    value_loss           | 0.215       |
-----------------------------------------
```


## Evaluation

To evaluate the results, run:

```
python run_eval_matrix.py --gpu 2
```

This script supports 3 modes:
1. AR
2. Naive EAGLE
3. EAGLE with depth and size policy

Default datasets are `gsm8k`, `mt_bench`, `alpaca`.

Resume / rerun behavior:
- If output is complete (line count >= question count), it is skipped.
- If output is partial, the file is deleted and rerun from scratch.

Useful examples:

```bash
# all 3 modes on default datasets
python run_eval_matrix.py --gpu 2

# only AR and Naive EAGLE
python run_eval_matrix.py --gpu 2 --modes ar naive-eagle

# run with specific policy checkpoints
python run_eval_matrix.py \
  --gpu 2 \
  --modes policy \
  --depth-policy checkpoints/depth/ppo_speculative_decoder_controller_step_420000.zip \
  --size-policy checkpoints/size/ppo_speculative_decoder_controller_step_50000.zip
```

Important arguments:
- `--datasets`: choose datasets to run.
- `--modes`: choose from `ar`, `naive-eagle`, `policy`.
- `--skip-completed/--no-skip-completed`: control skip behavior.
- `--rerun-partial/--no-rerun-partial`: control partial-file rerun behavior.

# Acknowledgements

This project has been influenced by many excellent projects in the LLM community, such as [EAGLE](https://github.com/SafeAILab/EAGLE/tree/main), [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), and others. 
