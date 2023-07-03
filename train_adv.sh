for confoun_lamb in 0.8 0.5 0.1
do
for lambda_impression in 0.1 0.01 0.001
do
  for confoun in True False
  do
    for down in MLP gmfBPR
    do
      for debias in True
      do
          python train_adv.py \
              --debias_mode Adversarial \
              --is_debias $debias \
              --confounder $confoun \
              --lambda_confounder $confoun_lamb \
              --downstream $down \
              --epoch_max 150 \
              --dataset pcic \
              --user_size 1000 \
              --item_size 1720 \
              --user_item_size 1000 1720 \
              --item_dim 1 \
              --user_dim 1 \
              --user_emb_dim 32 \
              --item_emb_dim 32 \
              --ipm_layer_dims 64 32 8 \
              --ctr_layer_dims 64 32 8 \
              --iter_save 10 \
              --clip_value 0.2 0.8 \
              --ctr_classweight 1 1 \
              --embedding_classweight 1 1 \
              --lambda_impression $lambda_impression
      done
    done
  done
done
done

