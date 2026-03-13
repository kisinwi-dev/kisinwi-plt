# Вклад в проект

## Наименование сервисов:

* **dms** - сервис(*data_manipulation_service*) отвечающий за работу с файлами.

## Наименование веток
- `<sercice>/feature/<feature-name>` — новая функция
- `<sercice>/fix/<bug-name>` — исправление ошибок
- `<sercice>/hotfix/<issue>` — экстренное исправление ошибок
- `<sercice>/chore/<task>` — рутинный занятия
- `<sercice>/refactor/<module>` — проверка кода

## Соглашение об описании в коммитах

- `feat(module): short description`
- `fix(module): short description`
- `docs(module): short description` — добавление документации
- `chore(module): short description`
- `refactor(module): short description`

### Примеры:

```
feat(dms): implement create dataset endpoint
```

## Правила написания PR
- Связывать PR c открытым *issue* (если такой имеется)
- PR описание должно включать:
  - Что было сделано
  - Как протестировать(на данный момент это файл **test.py** лежащий в сервисе)
  - Связка с *issue* (Closes #, Related to #)

## Дополнение

Правила в дальнейшем могут претерпевать изменения.
