FROM node:20-alpine as builder

WORKDIR /app

COPY yarn.lock /app/yarn.lock
RUN yarn install

COPY . /app/
RUN yarn build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html