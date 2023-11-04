FROM python:3.10.0 AS builder

RUN pip install --upgrade pip && pip install tensorflowjs

WORKDIR /models
COPY . /models
RUN mkdir /exports
RUN for model in $(ls -d */); do \
    mkdir /exports/$model; \
    rm -Rf exports/$model*; \
    tensorflowjs_converter \
    /models/$model \
    /exports/$model \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    --quantize_float16; \
    done

FROM nginx:alpine
COPY nginx.conf /etc/nginx/nginx.conf
COPY --from=builder /exports /usr/share/nginx/models

