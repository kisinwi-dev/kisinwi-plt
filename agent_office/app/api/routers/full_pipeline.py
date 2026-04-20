from fastapi import APIRouter, Query, HTTPException
from app.core.ml.crews import run_search_params_json
from app.core.analytic.crews import run_analysis
import requests
import json
import os

routers = APIRouter()

TASKER_URL = "http://" + os.getenv("TASKER_DOMEN", "localhost:6110")

def post_in_task(json_data):
    """Отправить JSON для запуска тренировки модели"""
    try:
        # Проверяем, что json_data не пустой
        if not json_data:
            return {"error": "JSON data is empty"}
        
        # Парсим JSON если это строка
        if isinstance(json_data, str):
            payload = json.loads(json_data)
        else:
            payload = json_data

        payload = {"payload": payload}

        # Отправляем POST запрос
        response = requests.post(
            f"{TASKER_URL}/tasks",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        # Проверяем статус ответа
        print(response.raise_for_status())
        
        return {
            "status": "success",
            "status_code": response.status_code,
            "response": response.json() if response.text else {"message": "Task created"}
        }
        
    except requests.exceptions.ConnectionError:
        return {"error": f"Не удалось подключиться к {TASKER_URL}. Сервис задач не запущен?"}
    except requests.exceptions.Timeout:
        return {"error": "Таймаут при отправке запроса"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP ошибка: {e.response.status_code}", "details": e.response.text}
    except Exception as e:
        return {"error": f"Ошибка: {str(e)}"}


@routers.get("/run_")
def health_status(
    dataset_id: str = Query(..., description="ID датасета для анализа"),
    version_id: str = Query(None, description="ID версии датасета"),
):
    """
    Анализ датасета и отправка задачи в сервис обучения.
    """
    try:
        # Шаг 1: Анализ датасета
        result_analysis = run_analysis(dataset_id, version_id)
        
        # Шаг 2: Подготовка JSON для задачи
        result_json = run_search_params_json(
            role="CV-enginer",
            previous_output=result_analysis.raw if hasattr(result_analysis, 'raw') else str(result_analysis)
        )
        
        # Получаем JSON строку
        json_to_send = result_json.raw if hasattr(result_json, 'raw') else str(result_json)
        
        # Шаг 3: Отправка в сервис задач
        post_result = post_in_task(json_to_send)
        
        # Шаг 4: Возвращаем результат
        return {
            "status": "completed",
            "dataset_id": dataset_id,
            "version_id": version_id,
            "analysis_completed": True,
            "json_prepared": True,
            "task_submission": post_result,
            "prepared_json": json.loads(json_to_send) if isinstance(json_to_send, str) else json_to_send
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при выполнении: {str(e)}"
        )
