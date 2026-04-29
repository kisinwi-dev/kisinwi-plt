from typing import Optional
from pymongo.errors import PyMongoError

from .storebase import ManagerBase
from app.api.schemes import *
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
                logger.warning(f"⚠️ Ответ с response_id {response.response_id} уже существует")
                return False
            
            result = self.collection.insert_one(response.model_dump())
            
            if result.inserted_id:
                logger.debug(
                    f"➕ Добавлен ответ агента {response.agent.name} "
                    f"для диалога {response.conversation_id}"
                )
                return True
            return False
            
        except PyMongoError as e:
            logger.error(f"😡 Ошибка добавления ответа: {e}")
            return False

    def get_conversation_responses(
            self, 
            conversation_id: str
    ) -> List[AgentResponse]:
        """Получение всех ответов для конкретного диалога"""
        try:
            cursor = self.collection.find(
                {'conversation_id': conversation_id}
            )
            
            responses = []
            for doc in cursor:
                doc.pop('_id', None)
                responses.append(AgentResponse(**doc))
            
            logger.debug(f"📖 Получено {len(responses)} ответов для диалога {conversation_id}")
            return responses
            
        except PyMongoError as e:
            logger.error(f"😡 Ошибка получения ответов диалога: {e}")
            return []

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
            return None
            
        except PyMongoError as e:
            logger.error(f"😡 Ошибка получения ответа: {e}")
            return None

    def delete_conversation(
            self, 
            conversation_id: str
    ) -> bool:
        """Удаление всех ответов диалога"""
        try:
            # Проверка существования
            exists = self.conversation_exists(conversation_id)
            
            if not exists:
                logger.warning(f"⚠️ Диалог {conversation_id} не найден")
                return False
            
            # Удаляем ответы
            self.collection.delete_many({'conversation_id': conversation_id})
            
            logger.debug(f"Диалог {conversation_id} удалён")
            return True
                
        except PyMongoError as e:
            logger.error(f"😡 Ошибка удаления диалога: {e}")
            return False

    def delete_response(
            self, 
            response_id: str
    ) -> bool:
        """Удаление ответа"""
        try:
            result = self.collection.delete_one({'response_id': response_id})
            
            if result.deleted_count > 0:
                logger.debug(f"🗑️ Удален ответ {response_id}")
                return True
            else:
                logger.debug(f"⚠️ Ответ {response_id} не найден")
                return False
                
        except PyMongoError as e:
            logger.error(f"😡 Ошибка удаления ответа: {e}")
            return False

    def conversation_exists(
            self, 
            conversation_id: str
    ) -> bool:
        """Проверка существования диалога"""
        try:
            result = self.collection.find_one(
                {'conversation_id': conversation_id},
                {'_id': 1}
            )
            return result is not None
            
        except PyMongoError as e:
            logger.error(f"😡 Ошибка проверки диалога: {e}")
            return False