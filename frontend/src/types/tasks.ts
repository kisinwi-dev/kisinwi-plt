// Типы сервиса задач обучения (tasker, порт 6110).

// Статусы задачи: waiting | running | completed | failed | cancelled.
export type TrainingTaskStatus = string;

export interface TrainingTask {
  id: string;
  name: string;
  model_id: string;
  discussion_id: string | null;
  agent_respons_ids: string[];
  status_id: number;
  status: TrainingTaskStatus;
  status_description: string;
  percentages: number;
  status_info: string | null;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  updated_at: string | null;
  completed_at: string | null;
}

export interface TrainingTasks {
  tasks: TrainingTask[];
}

export interface TasksQuery {
  status?: string;
  model_id?: string;
}
