python create_pretraining_data.py \
  --input_file=/data/wikitext-103-raw/wiki.train.raw \
  --output_file=./tmp/tf_examples.tfrecord \
  --vocab_file=./model/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5

python run_pretraining.py \
  --input_file=./tmp/tf_examples.tfrecord \
  --output_dir=./tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=./model/bert_config.json \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
