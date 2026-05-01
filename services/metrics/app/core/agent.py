from typing import Optional
from pymongo.errors import PyMongoError

from .mongo import ManagerBase
from app.api.schemes import AgentResponse
from app.logs import get_logger

logger = get_logger(__name__)

class AgentsResponseManager(ManagerBase):

    def add_response(
        self, 
        response: AgentResponse
    ) -> bool:
        """Добавление нового ответа агента"""
        try:
            # Проверяем, существования response
            existing = self.collection.find_one(
                {'response_id': response.response_id},
                {'_id': 1}
            )
            
            if existing:
                logger.warning(f"Метрики для ответа (id:'{response.response_id}') уже существуют")
                return False
            
            result = self.collection.insert_one(response.model_dump())
            
            if result.inserted_id:
                logger.debug(f"✅ Добавлены метрики ответа(id:'{response.response_id}')")
                return True
            return False
            
        except PyMongoError as e:
            logger.error(f"Ошибка добавления метрик ответа(id:'{response.response_id}'): {e}")
            return False

    def get_response_by_id(
            self, 
            response_id: str
    ) -> Optional[AgentResponse]:
        """Получение ответа по ID"""
        try:
            doc = self.collection.find_one({'response_id': response_id})
            
            if doc:
                doc.pop('_id', None)
                return AgentResponse(**doc)
            raise ValueError(f"Не найдены метрики ответа(id:'{response_id}')")
            
        except PyMongoError as e:
            logger.error(f"Ошибка получения ответа(id:'{response_id}'): {e}")
            return None

    def delete_response(
            self, 
            response_id: str
    ) -> bool:
        """Удаление ответа"""
        try:
            result = self.collection.delete_one({'response_id': response_id})
            
            if result.deleted_count > 0:
                logger.debug(f"Удален метрик ответа(id:'{response_id}')")
                return True
            else:
                logger.debug(f"Метрики ответа(id:'{response_id}') не найден")
                return False
                
        except PyMongoError as e:
            logger.error(f"Ошибка удаления метрик ответа(id:'{response_id}'): {e}")
            return False

    def response_exists(
            self, 
            response_id: str
    ) -> bool:
        """Проверка существования метрик ответа"""
        try:
            result = self.collection.find_one(
                {'response_id': response_id},
                {'_id': 1}
            )
            return result is not None
            
        except PyMongoError as e:
            logger.error(f"Ошибка проверки метрик ответа(id:'{response_id}'): {e}")
            return False