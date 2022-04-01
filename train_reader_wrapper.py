import os

def main():
    train_data = "open_domain_data/NQ/dpr-hard/entropy_sampling/train.json"
    eval_data = "open_domain_data/NQ/dpr-hard/dev.json"
    num_gpus = 8
    CUDA_VISIBLE_DEVICES = ",".join(str(i) for i in range(num_gpus))
    model_size = "small"
    per_gpu_batch_size = 2
    accumulation_steps = 4
    n_context = 20
    checkpoint_dir = f"checkpoint/nq_t5-{model_size}_dpr-hard_entropy/"
    min_steps = 10001
    max_steps = 40001
    num_subsets = 5
    num_times_to_eval = 6

    effective_batch_size = per_gpu_batch_size*accumulation_steps*num_gpus
    print(f"Effective Batch Size: {effective_batch_size}")

    for i in range(num_subsets):
        percent_of_data = int(100*(i+1)/num_subsets)
        total_steps = int(min_steps + (max_steps-min_steps)*percent_of_data/100)
        # Subtract 1 b/c if `eval_freq` is equal to `total_steps`, we never eval
        eval_freq = int(total_steps/num_times_to_eval) - 1

        cmd = f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} NGPU={num_gpus} python -m " \
              f"torch.distributed.launch --nproc_per_node={num_gpus} train_reader.py " \
              f"--train_data {train_data} " \
              f"--eval_data {eval_data} " \
              f"--model_size {model_size} " \
              f"--per_gpu_batch_size {per_gpu_batch_size} " \
              f"--accumulation_steps {accumulation_steps} " \
              f"--n_context {n_context} " \
              f"--checkpoint_dir {checkpoint_dir} " \
              f"--name {percent_of_data}% " \
              f"--total_steps {total_steps} " \
              f"--eval_freq {eval_freq} " \
              f"--percent_of_data {percent_of_data}"

        print(cmd, '\n')
        os.system(cmd)


if __name__ == "__main__":
    main()
