from typing import List, Optional
from pymongo.errors import PyMongoError

from .mongo import ManagerBase
from app.api.schemas import AgentResponse, AgentDiscussionMetrics
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

            logger.warning(f"Не найдены метрики ответа(id:'{response_id}')")
            return None

        except PyMongoError as e:
            logger.error(f"Ошибка получения ответа(id:'{response_id}'): {e}")
            return None

    def get_discussion_metrics(
            self,
            discussion_id: str
    ) -> AgentDiscussionMetrics:
        """Метрики всех агентов дискуссии и суммарная сводка по числовым полям"""
        responses: List[AgentResponse] = []
        try:
            for doc in self.collection.find({'discussion_id': discussion_id}):
                doc.pop('_id', None)
                responses.append(AgentResponse(**doc))
        except PyMongoError as e:
            logger.error(f"Ошибка получения метрик дискуссии(id:'{discussion_id}'): {e}")

        summary: dict = {"responses_count": len(responses)}
        for response in responses:
            for key, value in response.metrics.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                summary[key] = summary.get(key, 0) + value

        return AgentDiscussionMetrics(
            discussion_id=discussion_id,
            responses=responses,
            summary=summary,
        )

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