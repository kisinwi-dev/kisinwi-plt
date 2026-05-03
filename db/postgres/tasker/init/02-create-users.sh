set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '${APP_USER}') THEN
            CREATE USER ${APP_USER} WITH 
                PASSWORD '${APP_PASSWORD}'
                NOSUPERUSER
                NOCREATEDB
                NOCREATEROLE
                NOREPLICATION
                LOGIN;
            RAISE NOTICE '✅ Пользователь ${APP_USER} создан';
        ELSE
            RAISE NOTICE '⚠️ Пользователь ${APP_USER} уже существует';
        END IF;
    END
    \$\$;

    GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO ${APP_USER};
    GRANT USAGE ON SCHEMA public TO ${APP_USER};
    GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO ${APP_USER};
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO ${APP_USER};
    REVOKE CREATE ON SCHEMA public FROM ${APP_USER};
EOSQL

echo "✅ Пользователь приложения создан и настроен"