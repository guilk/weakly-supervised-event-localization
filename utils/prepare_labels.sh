python prepare_labels.py  --input_json=../../densevid/train.json \
    --max_length 30 \
    --output_json ../../densevid/activitynet_caption_vocab.json \
    --train_json ../../densevid/train.json \
    --val1_json  ../../densevid/val_1.json \
    --val2_json  ../../densevid/val_2.json
