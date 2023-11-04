FROM python:3.10.0 AS builder

RUN pip install --upgrade pip && pip install tensorflowjs

WORKDIR /models
COPY . /models
RUN mkdir /exports
RUN for model in $(ls -d */); do \
    echo $model; \
    mkdir /exports/$model; \
    rm -Rf exports/$model*; \
    tensorflowjs_converter \
    /models/$model \
    /exports/$model \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    --quantize_float16; \
    done

# # Add lockfile and package.json's of isolated subworkspace
# FROM node:20-alpine AS installer
# # Install python/pip
# ENV PYTHONUNBUFFERED=1
# RUN apk add --update --no-cache python3 && ln -sf python3 /usr/bin/python
# RUN python3 -m ensurepip
# RUN pip3 install --no-cache --upgrade pip setuptools

# RUN apk add --update --no-cache \
#   libc6-compat git curl pkgconfig \
#   make \
#   g++ \
#   jpeg-dev \
#   cairo-dev \
#   giflib-dev \
#   pango-dev \
#   libtool \
#   autoconf \
#   automake

# RUN apk update
# RUN npm i -g pnpm@8
# WORKDIR /app

# ARG YARN_BUILD_ENV
# ARG COMMIT_SHA

# ENV COMMIT_SHA $COMMIT_SHA

# # First install the dependencies (as they change less often)
# COPY .gitignore .gitignore
# COPY --from=builder /app/out/json/ .
# COPY --from=builder /app/out/pnpm-lock.yaml ./pnpm-lock.yaml
# RUN pnpm install

# # Build the project
# COPY --from=builder /app/out/full/ .
# COPY turbo.json turbo.json
# RUN pnpm turbo run build:$YARN_BUILD_ENV --filter=crm...

# FROM nginx:alpine

# COPY --from=builder /app/infra/crm/build/nginx.conf /etc/nginx/conf.d/default.conf
# COPY --from=installer /app/apps/crm/build /usr/share/nginx/html
