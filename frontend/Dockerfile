# Стадия для разработки: vite dev-сервер с hot-reload.
# Код монтируется volume'ом в docker-compose.dev.yml, образ нужен лишь для
# node-окружения и установленных зависимостей.
FROM node:20-alpine AS dev

WORKDIR /app

COPY package*.json ./

RUN npm ci

COPY . .

CMD ["npm", "run", "dev"]


FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./

RUN npm ci

COPY . .
ARG DATASETS_DOMAIN
ENV VITE_DMS=$DATASETS_DOMAIN
ARG AGENT_HISTORY_DOMAIN
ENV VITE_AGENT_HISTORY=$AGENT_HISTORY_DOMAIN

RUN npm run build

FROM nginx:alpine

COPY nginx.conf /etc/nginx/templates/default.conf.template
COPY --from=builder /app/dist /usr/share/nginx/html

ARG FRONTEND_PORT
EXPOSE ${FRONTEND_PORT}

CMD ["nginx", "-g", "daemon off;"]