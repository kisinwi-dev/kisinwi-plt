from typing import List, Optional
from pymongo.errors import PyMongoError

from .mongo import ManagerBase
from app.api.schemas import AgentResponse, AgentDiscussionMetrics
from app.logs import get_logger

logger = get_logger(__name__)

class AgentsResponseManager(ManagerBase):

    def ensure_indexes(self):
        """Уникальный индекс по response_id: защита от дубликатов и ускорение поиска"""
        try:
            self.collection.create_index('response_id', unique=True)
        except PyMongoError as e:
            logger.error(f"Не удалось создать индекс response_id: {e}")

    def add_response(
        self,
        response: AgentResponse
    ) -> bool:
        """Добавление нового ответа агента; False — метрики ответа уже существуют"""
        # Проверяем, существования response
        existing = self.collection.find_one(
            {'response_id': response.response_id},
            {'_id': 1}
        )

        if existing:
            logger.warning(f"Метрики для ответа (id:'{response.response_id}') уже существуют")
            return False

        self.collection.insert_one(response.model_dump())
        logger.debug(f"✅ Добавлены метрики ответа(id:'{response.response_id}')")
        return True

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
        for doc in self.collection.find({'discussion_id': discussion_id}):
            doc.pop('_id', None)
            responses.append(AgentResponse(**doc))

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