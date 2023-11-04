distination="models/jsmodel"
source="models/base-models"

# tensorflowjs_converter \
#     $source \
#     $distination \
#     --input_format=tf_saved_model \
#     --output_format=tfjs_graph_model \
#     --quantize_float16

for model in $(ls -d $source/*/ | grab -o '/([^/]+)/$' -r 1); do \
    echo $model; \
    mkdir /exports/$model; \
    # rm -Rf exports/$model*; \
    # tensorflowjs_converter \
    # /models/$model \
    # /exports/$model \
    # --input_format=tf_saved_model \
    # --output_format=tfjs_graph_model \
    # --quantize_float16; \
    done
