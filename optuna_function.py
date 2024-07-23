from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from training_setup import DataCollatorSpeechSeq2SeqWithPadding, compute_metrics
from dataset_preparation import load_and_prepare_dataset

def objective(trial, config):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [4, 8, 16])

    config['training_args']['learning_rate'] = learning_rate
    config['training_args']['per_device_train_batch_size'] = per_device_train_batch_size

    model, feature_extractor, tokenizer, processor = initialize_model(config['model_name'])
    train_test_split = load_and_prepare_dataset(config['metadata_path'], feature_extractor, tokenizer)

    training_args = Seq2SeqTrainingArguments(**config['training_args'])
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_test_split['train'],
        eval_dataset=train_test_split['test'],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer)
    )

    trainer.train()
    eval_result = trainer.evaluate(language="ca")

    return eval_result["eval_wer"]
