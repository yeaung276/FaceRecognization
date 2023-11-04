distination="models/jsmodel/mobile-net"
source="models/base-models/mobile-net"

tensorflowjs_converter \
    $source \
    $distination \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    --quantize_float16
