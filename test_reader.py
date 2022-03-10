# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import sys, os
import torch
import transformers
import numpy as np
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

import src.slurm
import src.util
from src.options import Options
import src.data
import src.evaluation
import src.model

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    f_x = exp_x / np.sum(exp_x)
    return f_x

def evaluate(model, dataset, dataloader, tokenizer, opt):
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    if opt.write_crossattention_scores:
        model.overwrite_forward_crossattention()
        model.reset_score_storage() 
    total = 0
    exactmatch = []
    answers, seq_scores = [], []
    if opt.write_results:
        write_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        fw = open(write_path / ('%d.txt'%opt.global_rank), 'a')
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            (idx, _, _, context_ids, context_mask) = batch

            if opt.write_crossattention_scores:
                model.reset_score_storage()

            # TODO: Swap these lines for commented out `.cuda()` lines.
            outputs = model.generate(
                # input_ids=context_ids,
                # attention_mask=context_mask,
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=50,
            )
            gen_sequence = outputs["sequences"]
            seq_score = outputs["scores"]
            if opt.write_crossattention_scores:
                # TODO: Swap these lines for commented out `.cuda()` lines.
                crossattention_scores = model.get_crossattention_scores(context_mask)
                # crossattention_scores = model.get_crossattention_scores(context_mask.cuda())

            for k, o in enumerate(gen_sequence):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                example = dataset.data[idx[k]]
                if 'answers' in example:
                    score = src.evaluation.ems(ans, example['answers'])
                    exactmatch.append(score)

                answers.append(ans)
                seq_scores.append(seq_score)

                if opt.write_results:
                    fw.write(str(example['id']) + "\t" + ans + '\n')
                if opt.write_crossattention_scores:
                    for j in range(context_ids.size(1)):
                        example['ctxs'][j]['score'] = crossattention_scores[k, j].item()

                total += 1
            if (i + 1) % opt.eval_print_freq == 0:
                log = f'Process rank:{opt.global_rank}, {i+1} / {len(dataloader)}'
                if len(exactmatch) == 0:
                    log += '| no answer to compute scores'
                else:
                    log += f' | average = {np.mean(exactmatch):.3f}'
                logger.warning(log)
    logger.warning(f'Process rank:{opt.global_rank}, total {total} | average = {np.mean(exactmatch):.3f}')
    if opt.is_distributed:
        torch.distributed.barrier()
    score, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    
    return {
        "answers": answers,
        "seq_scores": seq_scores,
        "score": score,
        "total": total,
        "exact_matches": exactmatch,
    }

def write_outputs_json(
    eval_dataset, 
    answers, 
    seq_scores, 
    # crossattention_scores,
    exact_matches, 
    outpath,
):
    """
    NB: Note that `exact_matches` uses dummy answers and is always 1, if dataset does not come
    with `targets`.
    """
    assert len(eval_dataset) == len(answers) == len(seq_scores)
    outputs = []
    for i in range(len(eval_dataset)):
        outputs.append({
            "question": eval_dataset.data[i]["question"],
            "ctxs": eval_dataset.data[i]["ctxs"][:eval_dataset.n_context],
            "answers": eval_dataset.data[i]["answers"],
            "predicted_answer": {
                "text": answers[i],
                "seq_scores": [float(np.max(softmax(ssv.cpu().numpy()[0]))) for ssv in
                               seq_scores[i]],
            },
        })
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(json.dumps(outputs, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    dir_path = Path(opt.checkpoint_dir)/opt.name
    directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if opt.write_results:
        (dir_path / 'test_results').mkdir(parents=True, exist_ok=True)
    logger = src.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')
    if not directory_exists and opt.is_main:
        options.print_options(opt)

    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base', return_dict=False)

    collator_function = src.data.Collator(opt.text_maxlength, tokenizer)
    eval_examples = src.data.load_data(
        opt.eval_data, 
        # global_rank=opt.global_rank, #use the global rank and world size attibutes to split the eval set on multiple gpus
        # world_size=opt.world_size
    )
    # NB: Added this code to deal with stupid target logic.
    if "target" not in eval_examples[0]:
        eval_examples = [{**ex, **{"target": "target"}} for ex in eval_examples]
    eval_dataset = src.data.Dataset(
        eval_examples, 
        opt.n_context, 
    )

    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=opt.per_gpu_batch_size,
        num_workers=0,
        collate_fn=collator_function
    )
    model_class = src.model.FiDT5
    model = model_class.from_pretrained(opt.model_path)
    model = model.to(opt.device)

    logger.info("Start eval")
    eval_outputs = evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)
    answers, seq_scores, exactmatch, total, ems = \
        eval_outputs["answers"], eval_outputs["seq_scores"], eval_outputs["score"], \
        eval_outputs["total"], eval_outputs["exact_matches"],

    logger.info(f'EM {100*exactmatch:.2f}, Total number of example {total}')

    # Save our scores out here:
    write_outputs_json(
        eval_dataset,
        answers,
        seq_scores,
        ems,
        os.path.join(opt.checkpoint_dir, opt.name, "predictions.json"),
    )

    if opt.write_results and opt.is_main:
        glob_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        write_path = Path(opt.checkpoint_dir) / opt.name / 'final_output.txt'
        src.util.write_output(glob_path, write_path)
    if opt.write_crossattention_scores:
        src.util.save_distributed_dataset(eval_dataset.data, opt)
