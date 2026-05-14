
# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Список папок для удаления
FOLDERS_TO_DELETE=(
    "db/agent_history/discussion"
    "db/mongodb/data"
    "db/postgres/ml_models/data"
    "db/postgres/tasker/data"
)

# Функция для вывода сообщений
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Проверка на запуск с sudo
check_sudo() {
    if [ "$EUID" -ne 0 ]; then 
        print_error "Скрипт должен запускаться с sudo!"
        echo "Использование: sudo $0"
        exit 1
    fi
}

# Функция удаления папки
delete_folder() {
    local folder_path="$1"
    
    if [ ! -d "$folder_path" ]; then
        print_warning "Папка не существует: $folder_path"
        return 0
    fi
    
    # Подсчёт количества файлов для информации
    local file_count=$(find "$folder_path" -type f 2>/dev/null | wc -l)
    
    print_info "Удаление: $folder_path ($file_count файлов)"
    
    if rm -rf "$folder_path"; then
        print_success "Удалено: $folder_path"
        return 0
    else
        print_error "Не удалось удалить: $folder_path"
        return 1
    fi
}

# Функция создания папок (с правильными правами)
create_folders() {
    print_info "Создание пустых папок с правильными правами..."
    
    # Создаём родительские папки
    mkdir -p db/agent_history 2>/dev/null
    mkdir -p db/mongodb 2>/dev/null
    mkdir -p db/postgres/ml_models 2>/dev/null
    mkdir -p db/postgres/tasker 2>/dev/null
    
    # Создаём пустые папки
    mkdir -p db/agent_history/discussion
    mkdir -p db/mongodb/data
    mkdir -p db/postgres/ml_models/data
    mkdir -p db/postgres/tasker/data
    
    # Устанавливаем права (777 для Docker, 755 для остальных)
    chmod 777 db/agent_history/discussion 2>/dev/null
    chmod 777 db/mongodb/data 2>/dev/null
    chmod 777 db/postgres/ml_models/data 2>/dev/null
    chmod 777 db/postgres/tasker/data 2>/dev/null
    
    print_success "Пустые папки созданы"
}

# Показать что будет удалено
show_summary() {
    echo ""
    echo "========================================="
    print_info "Будут удалены следующие папки:"
    echo "========================================="
    for folder in "${FOLDERS_TO_DELETE[@]}"; do
        if [ -d "$folder" ]; then
            local size=$(du -sh "$folder" 2>/dev/null | cut -f1)
            echo "  - $folder ($size)"
        else
            echo "  - $folder (не существует)"
        fi
    done
    echo "========================================="
}

# Основная функция
main() {
    echo "========================================="
    echo "      🧹 ОЧИСТКА ДАННЫХ ПРОЕКТА"
    echo "========================================="
    
    # Проверка на sudo
    check_sudo
    
    # Показать что будет удалено
    show_summary
    
    # Запрос подтверждения
    echo ""
    read -p "Продолжить удаление? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Операция отменена"
        exit 0
    fi
    
    # Удаление папок
    echo ""
    print_info "Начало удаления..."
    echo ""
    
    local success_count=0
    local fail_count=0
    
    for folder in "${FOLDERS_TO_DELETE[@]}"; do
        if delete_folder "$folder"; then
            ((success_count++))
        else
            ((fail_count++))
        fi
    done
    
    # Создание пустых папок
    echo ""
    print_info "Создание пустых папок..."
    create_folders
    
    # Итог
    echo ""
    echo "========================================="
    print_success "Очистка завершена!"
    echo "✅ Удалено папок: $success_count"
    if [ $fail_count -gt 0 ]; then
        echo "❌ Не удалось удалить: $fail_count"
    fi
    echo "📁 Созданы новые пустые папки"
    echo "========================================="
}

# Запуск
main "$@"