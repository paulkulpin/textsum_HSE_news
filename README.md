# Using Automatic Text Reduction Techniques for Annotating News Portal Articles.

## Abstract
Automatic text reduction, a.k.a. text summarization, is one of the most common sequence-to-sequence tasks in the field of Natural Language Processing (NLP). It is a challenging problem that involves reducing the length of a given text while retaining its most important information. This work reviews various methods for text summarization, including extractive and abstractive approaches, and provides an overview of different deep learning architectures that have been applied to this task, as well as user interface implementation tools and backend technologies. The work then focuses on the training of several neural network models for Russian texts, using a dataset of news articles from the High School of Economics news platform, and analyzing their results with different text quality metrics. The final stage of the work involves deploying the resulting text summarization system to the university’s website.

Full dimploma: [in Russian only](https://github.com/paulkulpin/textsum_HSE_news/blob/main/TextSum_HSEnews_diploma.pdf)

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
    --path_to_json <path> \
    --path_to_csv <path> \
    --source_text_field_name body \
    --annotation_field_name annotation \
    --max_document_char_length 15000 \
    --min_document_char_length 100 \
    --max_summary_char_length 1500 \
    --min_summary_char_length 10 \
    --delete_stop_words False 
```


### Training:
```bash
python scripts/main.py \
    --action training \ #action to
    --model_type FRED \
    --csv_dataset_path <path> \
    --shuffle_dataset True \
    --reduce_dataset 100.0 \
    --model_state_path <path> \
    --res_model_state_path <path> \
    --res_model_comments_path <path> \
    --HF_model_name FRED-T5-large \
    --random_seed 101 \
    --cuda_idx 0 \
    --num_workers 0 \
    --source_text_field_name document \
    --annotation_field_name summary \
    --need_wandb True \
    --wandb_key <key> \
    --wandb_project_name <name> \
    --wandb_run_name <name> \
    --accum_steps 1 \
    --use_mp False \
    --use_clipping False \
    --batch_size 2 \
    --num_epochs 2 \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --num_warmup_steps 1000 \
    --num_cycles 0.5 \
    --max_input_length 1000 \
    --max_target_length 200 \
    --ignore_warnings False \
    --dir_for_in_between_savings <path> \
    --saving_steps_fraction 0.1 
```

### Evaluating:
```bash
python scripts/main.py \
    --action evaluating \ #action to
    --model_type FRED \
    --csv_dataset_path <path> \
    --shuffle_dataset False \
    --reduce_dataset 100.0 \
    --model_state_path <path> \
    --HF_model_name FRED-T5-large \
    --random_seed 101 \
    --cuda_idx 0 \
    --num_workers 0 \
    --source_text_field_name document \
    --annotation_field_name summary \
    --need_wandb True \
    --wandb_key <key> \
    --wandb_project_name <name> \
    --wandb_run_name <name> \
    --batch_size 2 \
    --eval_max_length 150 \
    --eval_min_length 10 \
    --ignore_warnings False
```

### To get help:
```bash
python scripts/main.py -h
```



