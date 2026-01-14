from .filesystem import FileSystemManager
from .models import ClassInfo, VersionInfo, DatasetInfo

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

    def get_dataset_version_name(self, dataset_name: str) -> list[str]:
        self._fsm.reset()
        self._fsm.in_dir(dataset_name)
        version_list = self._fsm.get_all_dir()
        return version_list        

    def get_dataset_classes_name(
            self, 
            dataset_name: str,
            dataset_version: str,
        ) -> list[str]:
        self._fsm.reset()
        self._fsm.in_dirs([dataset_name, dataset_version])
        classes_list = self._fsm.get_all_dir()
        return classes_list
    
    def get_dataset_files_name(
            self,
            dataset_name: str,
            dataset_version: str,
            class_name: str
        ) -> list[str]:
        self._fsm.reset()
        self._fsm.in_dirs([dataset_name, dataset_version, class_name])
        return self._fsm.get_all_file()
    
    def get_dataset_info(
            self,
            dataset_name: str
        ):
        self._fsm.reset()
        self._fsm.in_dir(dataset_name)
        versions = self._fsm.get_all_dir()
        info_versions = []
        for version in versions:
            self._fsm.in_dir(version)
            classes = self._fsm.get_all_dir()
            info_classes = []
            for class_ in classes:
                self._fsm.in_dir(class_)
                files = self._fsm.get_all_file()
                info_classes.append(
                    ClassInfo(
                        name=class_,
                        description=None,
                        count_files=len(files),
                        type_files=None
                    )
                )
                self._fsm.out_dir()
            
            info_versions.append(
                VersionInfo(
                    name=version,
                    description=None,
                    classes=info_classes
                )
            )
            self._fsm.out_dir()
        
        return DatasetInfo(
            name=dataset_name,
            description=None,
            versions=info_versions
        )

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
