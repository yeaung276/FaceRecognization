distination="web/public/jsmodel"
source="models/mobile_net"

mkdir $distination
rm -Rf $distination/*

tensorflowjs_converter \
    $source \
    $distination \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    --quantize_float16

