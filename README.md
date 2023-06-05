# Using Automatic Text Reduction Techniques for Annotating News Portal Articles

## Abstract
Automatic text reduction, a.k.a. text summarization, is one of the most common sequence-to-sequence tasks in the field of Natural Language Processing (NLP). It is a challenging problem that involves reducing the length of a given text while retaining its most important information. This work reviews various methods for text summarization, including extractive and abstractive approaches, and provides an overview of different deep learning architectures that have been applied to this task, as well as user interface implementation tools and backend technologies. The work then focuses on the training of several neural network models for Russian texts, using a dataset of news articles from the High School of Economics news platform, and analyzing their results with different text quality metrics. The final stage of the work involves deploying the resulting text summarization system to the university’s website.

## Keywords:
Deep learning, text summarization task, dataset, full-stack NLP application.

**Full diploma paper: [in Russian only](https://github.com/paulkulpin/textsum_HSE_news/blob/main/TextSum_HSE_news_diploma.pdf)**
---

## To run the backend file:

```bash
sudo docker build -t "sum_text_hse_back" -f service/backend_utils/dockerfile . 
```
```bash
sudo docker run --gpus all -v <model_states path>:/app/model_states -p 8000:8000 sum_text_hse_back
```

## To run the frontend file:
```bash
sudo docker build -t "sum_text_hse_front" -f service/frontend_utils/dockerfile . 
```
```bash
sudo docker run -p 8501:8501 sum_text_hse_front
```

## How to use scripts:
```bash
pip install -q -r scripts/requirements.txt
```

### Data processing:
```bash
python scripts/dataset_utils/dataprocess.py \
    --path_to_json <path> \ #   Full path to originally scrapped json file. (obligatory)
    --path_to_csv <path> \  #   Full path to final dataset's csv file. (obligatory)
    --source_text_field_name body \ #   Name of field that includes article.
    --annotation_field_name annotation \    #   Name of field that includes annotation.
    --max_document_char_length 15000 \  #   Acticles with symbols length more than <value> symbols will be filtered out.
    --min_document_char_length 100 \    #   Acticles with symbols length less than <value> symbols will be filtered out.
    --max_summary_char_length 1500 \    #   Annotation with symbols length more than <value> symbols will be filtered out.
    --min_summary_char_length 10 \  #   Annotation with symbols length less than <value> symbols will be filtered out.
    #--delete_stop_words True \  #   If stopwords deleting required. Default value is False. Do not use this parameter if you exactly want not to delete.
    --sum_doc_proportion 0.5 \  #   To delete all the examples with len(annotation)/len(article) > <value>.
    --timeout_time 2    #   If timeout_time == 0, there will be no timeout for each example. It may work much faster, but may stop during BeatufulSoup exception.
```


### Training:
```bash
python scripts/main.py \
    --action training \ #   Action to execute [training/evaluate]. (obligatory)
    --model_type FRED \ #   Model name [ruT5/ruGPT/mBART/FRED]. (obligatory)
    --csv_dataset_path <path> \ #   Full path to dataset csv file. (obligatory)
    --HF_model_name FRED-T5-large \ #   Hugging Face model/tokenizer name. (obligatory)
    --shuffle_dataset True \    #   If dataset shuffling needed? [True/False]
    --reduce_dataset 100.0 \    #   Percentage part of dataset to keep. [0.0 - 100.0]
    --model_state_path <path> \ #   Path to current model state_dict. If not used, script uses [HuggingFace](https://huggingface.co/ai-forever/FRED-T5-large) checkpoint.
    --res_model_state_path <path> \ #   Path to resulting model state_dict. If not used, model state will be saved in the current path. Only for training.
    --res_model_comments_path <path> \  #   Path to json file with comments for model_state.
    --random_seed 101 \ #   Random seed for torch, numpy.
    --cuda_idx 0 \  #   For several GPUs.
    --num_workers 0 \   #   Number of workers for dataloader, etc.
    --source_text_field_name document \ #   Name of field that includes article.
    --annotation_field_name summary \   #   Name of field that includes annotation.
    --need_wandb True \ #   If [Weights & Biases](https://wandb.ai/) logging needed.
    --wandb_key <key> \ #   Key for [Weights & Biases](https://wandb.ai/) logging. 
    --wandb_project_name <name> \
    --wandb_run_name <name> \
    --accum_steps 1 \   #   Number of steps for 1 scheduler step.
    --use_mp False \    #   If [Mixed Precision](https://pytorch.org/docs/stable/notes/amp_examples.html) needed.
    --use_clipping False \  #   If [clipping](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) needed.
    --batch_size 2 \
    --num_epochs 2 \    #   Number of epoches to train.
    --lr 0.0001 \   #   For AdamW optimization.
    --weight_decay 0.01 \   #   For AdamW optimization.
    --num_warmup_steps 1000 \   #   For Cosine Annealing Scheduler.
    --num_cycles 0.5 \  # For Cosine Annealing Scheduler.
    --max_input_length 1000 \   #   For tokenizer encoding.
    --max_target_length 200 \   #   For tokenizer encoding.
    --ignore_warnings False \
    --dir_for_in_between_savings <path> \
    --saving_steps_fraction 0.1 
```

### Evaluating:
```bash
python scripts/main.py \
    --action evaluating \   #   Action to execute [training/evaluate]. (obligatory)
    --model_type FRED \ #   Model name [ruT5/ruGPT/mBART/FRED]. (obligatory)
    --csv_dataset_path <path> \ #   Full path to dataset csv file. (obligatory)
    --HF_model_name FRED-T5-large \ #   Hugging Face model/tokenizer name. (obligatory)
    --shuffle_dataset False \   #   If dataset shuffling needed? [True/False]
    --reduce_dataset 100.0 \    #   Percentage part of dataset to keep. [0.0 - 100.0]
    --model_state_path <path> \ #   Path to current model state_dict. If not used, script uses [HuggingFace](https://huggingface.co/ai-forever/FRED-T5-large) checkpoint.
    --random_seed 101 \ #   Random seed for torch, numpy.
    --cuda_idx 0 \  #   For several GPUs.
    --num_workers 0 \   #   Number of epoches to train
    --source_text_field_name document \ #   Name of field that includes article.
    --annotation_field_name summary \   #   Name of field that includes annotation.
    --need_wandb True \ #   If [Weights & Biases](https://wandb.ai/) logging needed.
    --wandb_key <key> \ #   Key for [Weights & Biases](https://wandb.ai/) logging. 
    --wandb_project_name <name> \
    --wandb_run_name <name> \
    --batch_size 2 \
    --eval_max_length 150 \ #   Max tokens for generating.
    --eval_min_length 10 \  # Min tokens for generating.
    --ignore_warnings False
```

### To get help:
```bash
python scripts/main.py -h
```



