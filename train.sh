lr=1e-3
batch_size=1
lambda_style=1.0
lambda_content=1.0
model_path=lr_${lr}_bs_${batch_size}_style_${lambda_style}_content_${lambda_content}
ckpt_path=ckpts/${model_path}/
result_path=results/${model_path}/

python train.py --lr ${lr} --batch_size ${batch_size} --lambda_style ${lambda_style} \
 --lambda_content ${lambda_content} --ckpt_path ${ckpt_path} --result_path ${result_path}
