set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '${APP_TASKER_USER}') THEN
            CREATE USER ${APP_TASKER_USER} WITH 
                PASSWORD '${APP_TASKER_PASSWORD}'
                NOSUPERUSER
                NOCREATEDB
                NOCREATEROLE
                NOREPLICATION
                LOGIN;
            RAISE NOTICE '✅ Пользователь ${APP_TASKER_USER} создан';
        ELSE
            RAISE NOTICE '⚠️ Пользователь ${APP_TASKER_USER} уже существует';
        END IF;
    END
    \$\$;

    GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO ${APP_TASKER_USER};
    GRANT USAGE ON SCHEMA public TO ${APP_TASKER_USER};
    GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO ${APP_TASKER_USER};
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO ${APP_TASKER_USER};
    REVOKE CREATE ON SCHEMA public FROM ${APP_TASKER_USER};
EOSQL

echo "✅ Пользователь приложения создан и настроен"