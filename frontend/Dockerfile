FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./

RUN npm ci

COPY . .
ARG DATASETS_DOMAIN
ENV VITE_DMS=$DATASETS_DOMAIN

RUN npm run build

FROM nginx:alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 6001

CMD ["nginx", "-g", "daemon off;"]