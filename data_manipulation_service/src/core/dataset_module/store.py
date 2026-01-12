from core.dataset_module.filesystem import FileSystemManager

class Store:
    def __init__(
            self,
            fsm: FileSystemManager | None = None
        ):
        self._fsm = fsm if fsm else FileSystemManager()

    def get_dataset_name(self) -> list[str]:
        self._fsm.reset()
        dataset_list = self._fsm.get_all_dir()
        return dataset_list

    def get_dataset_version(self, dataset_name: str) -> list[str]:
        self._fsm.reset()
        self._fsm.in_dir(dataset_name)
        version_list = self._fsm.get_all_dir()
        return version_list        

    def get_dataset_classes(
            self, 
            dataset_name: str,
            dataset_version: str,
        ) -> list[str]:
        self._fsm.reset()
        self._fsm.in_dirs([dataset_name, dataset_version])
        classes_list = self._fsm.get_all_dir()
        return classes_list
    
    def get_dataset_files(
            self,
            dataset_name: str,
            dataset_version: str,
            class_name: str
        ) -> list[str]:
        self._fsm.reset()
        self._fsm.in_dirs([dataset_name, dataset_version, class_name])
        return self._fsm.get_all_file()
    
    def get_dataset_info(self):
        pass

    def set_new_dataset(self):
        pass

    def drop_dataset(self):
        pass

    def rename_dataset(self):
        pass

    def rename_dataset_version(self):
        pass

    def set_new_dataset_version(self):
        pass

    def validation_dataset(self):
        pass
