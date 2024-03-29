
export GVIRTUS_HOME=/opt/GVirtuS
export GVIRTUS_LOGLEVEL=0
export GVIRTUS_CONFIG=$GVIRTUS_HOME/etc/properties.json
export LD_LIBRARY_PATH=$GVIRTUS_HOME/lib:$LD_LIBRARY_PATH
export LD_PRELOAD="/opt/GVirtuS/lib/frontend/libcudart.so /opt/GVirtuS/lib/frontend/libcudnn.so /opt/GVirtuS/lib/frontend/libcufft.so /opt/GVirtuS/lib/frontend/libcurand.so"
# python3 -W ignore test.py
python3 -W ignore GPUoffload_test/inference.py --root_path data/sample_scannet \
--resize_size 256 --texture_size 4096,4096 \
--min_images 1 --max_images 1000 --scene scene0673_00_closeup \
--hierarchical --hierarchical_layers 4 \
--loss_weight content=7e1 \
--loss_weight style=1e-4 --style_weights="1000,1000,10,10,1000" \
--loss_weight tex_reg=5e3 \
--vgg_gatys_model_path data/vgg_conv.pth \
--renderer_mipmap path/to/git/neural-rendering-style-transfer/scripts/scannet/render_uv/build/scannet_uv_renderer \
--learning_rate 1 --decay_step_size 3 \
--log_images_nth 5000 --batch_size 1 \
--max_epochs 7 \
--train_split 0.99 --val_split 0.01 \
--sampler_mode repeat --index_repeat 20 \
--save_texture --split_mode sequential \
--num_workers 4 \
--style_image_path styles/120styles/14-2.jpg \
--style_pyramid_mode "multi" \
--gram_mode "current" \
--angle_threshold 30 \
--pyramid_levels 4 \
--min_pyramid_depth 0.25 \
--min_pyramid_height 256 \
--times 100