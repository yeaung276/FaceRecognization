distination="web/public/jsmodel"
source="models/mobile_net/saved_model.pb"

mkdir $distination
rm -Rf $distination/*

tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    --quantize_float16 \
    $source \
    $distination
