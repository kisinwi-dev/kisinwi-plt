# hf_cache

Кэш весов pretrained-моделей для сервиса trainer (bind-mount в `/app/hf_cache`).

- `huggingface/` - кэш `huggingface_hub` (`HF_HOME`), сюда timm качает веса с HF Hub.
- `torch/` - кэш `torch.hub` (`TORCH_HOME`) для legacy-моделей timm.

Содержимое не коммитится (см. `.gitignore`): веса качаются один раз и переживают пересоздание
контейнера. Каталог создаётся контейнером под root - застрявшие `*.incomplete` чистить через контейнер.
